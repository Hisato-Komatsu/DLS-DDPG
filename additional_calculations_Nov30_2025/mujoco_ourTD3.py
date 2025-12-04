# clculation of our TD3
#10/14 can choose whether to use initial DDPG update (expressed as start_steps) 

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["VECLIB_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import gymnasium as gym
import numpy as np
import random
import copy
from scipy.optimize import fmin_l_bfgs_b
from collections import namedtuple
from collections import deque

#from gym.wrappers import RecordVideo

import torch
from torch import nn
import torch.nn.functional as torF
from torch.distributions import Categorical

import time
start_time=time.time()

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(seed)


if torch.backends.openmp.is_available():
   torch.set_num_threads(1)


class hyperparameters:
   def __init__(self):
      #self.game = 'InvertedPendulum-v5'
      #self.game = 'InvertedDoublePendulum-v5'
      #self.game = 'HalfCheetah-v5'
      #self.game = 'Hopper-v5'
      #self.game = 'Walker2d-v5'
      self.game = 'Ant-v5'
   
      self.gamma = 0.99
      self.beta_Q = 1.0e-2 #1.0e-2
      self.beta_a = 1.0e-2 #1.0e-2
      self.beta_Q_min = 1.0e-3 #1.0e-3
      self.beta_a_min = 1.0e-3 #1.0e-3

      self.learning_rate = 1.0e-3
      #self.LS_batch_size = 10000
      self.layer_size = 1024
      self.memory_size = 1000000
      self.minibatch_size = 256

      self.sigma = 0.1
      self.tau = 0.005
      #self.tau_LS = 0.1

      self.TPS = True
      self.TPS_sigma = 0.2
      self.TPS_bound = 0.5
      self.policy_delay = 2
      
      """
      self.weight_actor = 2.0
      self.weight_critic = 2.0      
      self.restrict_upb = 0.4
      
      self.LS_actor = True
      self.LS_critic = True
      self.DDPG_actor = True
      self.DDPG_critic = True

      self.optim_choosing = True
      """
      self.resid_coef = 0.001
      
      self.random_steps = 25000
      
      self.movie_record = True
      #self.weight_record = False
      self.time_record = False
      
      self.start_steps = True


class Actor_Net(nn.Module):
   def __init__(self, env, layer_size):
      super().__init__()
      self.state_size = env.observation_space.shape[0] 
      self.action_size = env.action_space.shape[0]
      self.layer_size = layer_size

      self.fc1 = nn.Linear(self.state_size, self.layer_size)
      self.fc2 = nn.Linear(self.layer_size, self.action_size)

class Critic_Net(nn.Module):
   def __init__(self, env, layer_size):
      super().__init__()
      self.state_size = env.observation_space.shape[0] 
      self.action_size = env.action_space.shape[0]
      self.layer_size = layer_size

      self.fc1_s = nn.Linear(self.state_size, self.layer_size)
      self.fc1_a = nn.Linear(self.action_size, self.layer_size, bias=False)
      self.fc2 = nn.Linear(self.layer_size, 1)

class Whole_Net():
   def __init__(self, env, hyperparameters):
      super().__init__()
      self.hyperparameters = hyperparameters
      self.state_size = env.observation_space.shape[0] 
      self.action_size = env.action_space.shape[0]
      self.lower_bound = torch.tensor(env.action_space.low)
      self.upper_bound = torch.tensor(env.action_space.high)

      self.gamma = self.hyperparameters.gamma
      self.beta_Q = self.hyperparameters.beta_Q 
      self.beta_a = self.hyperparameters.beta_a 
      self.beta_Q_min = self.hyperparameters.beta_Q_min
      self.beta_a_min = self.hyperparameters.beta_a_min

      self.layer_size = self.hyperparameters.layer_size
      self.sigma = self.hyperparameters.sigma
      self.tau = self.hyperparameters.tau

      self.main_net_actor = Actor_Net(env, layer_size=self.layer_size)
      self.main_net_critic_A = Critic_Net(env, layer_size=self.layer_size)
      self.main_net_critic_B = Critic_Net(env, layer_size=self.layer_size)
      self.target_net_actor = copy.deepcopy(self.main_net_actor)
      self.target_net_critic_A = copy.deepcopy(self.main_net_critic_A)
      self.target_net_critic_B = copy.deepcopy(self.main_net_critic_B)

      self.memory_size = self.hyperparameters.memory_size
      self.minibatch_size = self.hyperparameters.minibatch_size
      self.memory = []
      self.memory_full = False

      self.optimizer_actor = torch.optim.Adam(list( self.main_net_actor.parameters() ), lr=hyper.learning_rate)
      self.optimizer_critic_A = torch.optim.Adam(list( self.main_net_critic_A.parameters() ), lr=hyper.learning_rate)
      self.optimizer_critic_B = torch.optim.Adam(list( self.main_net_critic_B.parameters() ), lr=hyper.learning_rate)

      self.TPS = self.hyperparameters.TPS
      self.TPS_sigma = self.hyperparameters.TPS_sigma
      self.TPS_bound = self.hyperparameters.TPS_bound
      self.policy_delay = self.hyperparameters.policy_delay
      self.critic_step = 1
      
      self.resid_coef = self.hyperparameters.resid_coef 

      self.W_out_a = np.zeros( (self.action_size,self.layer_size+1) )
      self.W_out_Q_A = np.zeros( (1,self.layer_size+1) )
      self.W_out_Q_B = np.zeros( (1,self.layer_size+1) )

   def forward_actor(self, x, add_rand=True, return_hidden=False, return_res=False):
      x = x.float()
      u = torch.flatten(x, 1) # flatten all dimensions except minibatch

      u = torF.tanh(self.main_net_actor.fc1(u))
      mu = self.main_net_actor.fc2(u)
      if add_rand:
         mu_rand = torch.normal(mean=torch.zeros(mu.shape), std=self.sigma*torch.ones(mu.shape))
         mu_clip = torch.clip( torch.clip( mu, self.lower_bound, self.upper_bound) + mu_rand, self.lower_bound, self.upper_bound )
      else:
         mu_clip = torch.clip( mu, self.lower_bound, self.upper_bound) 

      if return_res:
         resid = torch.mean( (mu - mu_clip)**2 )
         return mu_clip, resid
      else:
         return mu_clip

   def forward_critic_A(self, x, a):
      x = x.float()
      a = a.float()
      us = torch.flatten(x, 1)
      ua = torch.flatten(a, 1)
      
      u = torF.tanh(self.main_net_critic_A.fc1_s(us) + self.main_net_critic_A.fc1_a(ua))
      q = self.main_net_critic_A.fc2(u).squeeze(1)
      return q

   def forward_critic_B(self, x, a):
      x = x.float()
      a = a.float()
      us = torch.flatten(x, 1)
      ua = torch.flatten(a, 1)
      
      u = torF.tanh(self.main_net_critic_B.fc1_s(us) + self.main_net_critic_B.fc1_a(ua))
      q = self.main_net_critic_B.fc2(u).squeeze(1)
      return q

   def target_q(self, x, add_rand):
      x = x.float()
      ut0 = torch.flatten(x, 1) # flatten all dimensions except minibatch
      
      ut = torF.tanh(self.target_net_actor.fc1(ut0))
      mut = self.target_net_actor.fc2(ut)
      mut = torch.clip( mut, self.lower_bound, self.upper_bound)
      if add_rand:
         mut_rand = torch.normal(mean=torch.zeros(mut.shape), std=self.TPS_sigma*torch.ones(mut.shape))
         mut_rand = torch.clip(mut_rand, -self.TPS_bound, self.TPS_bound)
         mut = torch.clip( mut+mut_rand, self.lower_bound, self.upper_bound)
         
      #vt_0 = torch.cat([ut0,mut], dim=1)
      
      vt_A = torF.tanh(self.target_net_critic_A.fc1_s(ut0) + self.target_net_critic_A.fc1_a(mut))
      vt_B = torF.tanh(self.target_net_critic_B.fc1_s(ut0) + self.target_net_critic_B.fc1_a(mut))
      qt = torch.minimum( self.target_net_critic_A.fc2(vt_A).squeeze(1) , self.target_net_critic_B.fc2(vt_B).squeeze(1) )
      return qt

   def memory_append(self, obj):
      if not self.memory_full:
         self.memory.append(obj)
         if len(self.memory) >= self.memory_size :
            self.memory_full = True
            self.memory_position = 0
      else:
         self.memory[self.memory_position%self.memory_size] = obj
         self.memory_position += 1

   def size(self):
      return len(self.memory)

   def memory_sample(self, sample_size):
      current_size = self.size()
      if current_size < sample_size:
         minibatch = random.sample(self.memory, current_size)
      else:
         minibatch = random.sample(self.memory, sample_size)
      res = []
      for i in range(5):
         k = np.stack(tuple(item[i] for item in minibatch), axis=0)
         res.append(torch.tensor(k))
      return res[0], res[1], res[2], res[3], res[4]

   def TD3_train(self, train_actor=False, train_critic=False):

      s_sample, a_sample, r_sample, next_s_sample, done_sample = self.memory_sample(sample_size = self.minibatch_size)
      if train_critic:
         with torch.no_grad():
            target_q = self.target_q( next_s_sample, add_rand=self.TPS )
         target = r_sample + self.gamma*target_q*(1 - done_sample.float())
         
         main_q_critic_A = self.forward_critic_A( s_sample, a_sample )
         loss_critic_A = torF.mse_loss(main_q_critic_A.float(), target.float(), reduction='mean') 
         loss_critic_A = loss_critic_A + self.beta_Q*( torch.sum(self.main_net_critic_A.fc1_s.weight**2) + torch.sum(self.main_net_critic_A.fc1_s.bias**2) + torch.sum(self.main_net_critic_A.fc1_a.weight**2) + torch.sum(self.main_net_critic_A.fc2.weight**2) + torch.sum(self.main_net_critic_A.fc2.bias**2) ) 
         
         self.optimizer_critic_A.zero_grad()
         loss_critic_A.backward()
         #nn.utils.clip_grad_norm_(self.main_net_critic_A.parameters(), max_grad_norm)
         self.optimizer_critic_A.step()
         del loss_critic_A
                  
         main_q_critic_B = self.forward_critic_B( s_sample, a_sample )
         loss_critic_B = torF.mse_loss(main_q_critic_B.float(), target.float(), reduction='mean') 
         loss_critic_B = loss_critic_B + self.beta_Q*( torch.sum(self.main_net_critic_B.fc1_s.weight**2) + torch.sum(self.main_net_critic_B.fc1_s.bias**2) + torch.sum(self.main_net_critic_B.fc1_a.weight**2) + torch.sum(self.main_net_critic_B.fc2.weight**2) + torch.sum(self.main_net_critic_B.fc2.bias**2) ) 
         
         self.optimizer_critic_B.zero_grad()
         loss_critic_B.backward()
         #nn.utils.clip_grad_norm_(self.main_net_critic_B.parameters(), max_grad_norm)
         self.optimizer_critic_B.step()
         del loss_critic_B

      if self.critic_step%self.policy_delay == 0:
         if train_actor:
            mu_actor, resid = self.forward_actor( s_sample, add_rand=False, return_res=True ) 
            main_q_actor = self.forward_critic_A( s_sample, mu_actor ) #uses critic A 
            loss_actor = - torch.mean( main_q_actor ) 
            loss_actor = loss_actor + self.beta_a*( torch.sum(self.main_net_actor.fc1.weight**2) + torch.sum(self.main_net_actor.fc1.bias**2) + torch.sum(self.main_net_actor.fc2.weight**2) + torch.sum(self.main_net_actor.fc2.bias**2) ) 
            loss_actor = loss_actor + self.resid_coef*resid
         
            self.optimizer_actor.zero_grad()
            loss_actor.backward()
            #nn.utils.clip_grad_norm_(self.main_net_actor.parameters(), max_grad_norm)
            self.optimizer_actor.step()
         
            del loss_actor

         self.update_target(update_actor=train_actor, update_critic=train_critic)
      
      self.critic_step = (self.critic_step+1)%self.policy_delay

   def update_target(self, update_actor=False, update_critic=False):
      if update_actor:
         for target_param_a, param_a in zip(self.target_net_actor.parameters(), self.main_net_actor.parameters()):
            target_param_a.data.copy_( target_param_a.data * (1.0 - self.tau) + param_a.data * self.tau )
      if update_critic:
         for target_param_c_A, param_c_A in zip(self.target_net_critic_A.parameters(), self.main_net_critic_A.parameters()):
            target_param_c_A.data.copy_( target_param_c_A.data * (1.0 - self.tau) + param_c_A.data * self.tau )
         for target_param_c_B, param_c_B in zip(self.target_net_critic_B.parameters(), self.main_net_critic_B.parameters()):
            target_param_c_B.data.copy_( target_param_c_B.data * (1.0 - self.tau) + param_c_B.data * self.tau )

   def Ridge_decay(self):
      self.W_out_a[:,:self.layer_size] = self.main_net_actor.fc2.weight.detach().numpy().copy()
      self.W_out_a[:,-1] = self.main_net_actor.fc2.bias.detach().numpy().copy()
      self.W_out_Q_A[:,:self.layer_size] = self.main_net_critic_A.fc2.weight.detach().numpy().copy()
      self.W_out_Q_A[:,-1] = self.main_net_critic_A.fc2.bias.detach().numpy().copy()
      self.W_out_Q_B[:,:self.layer_size] = self.main_net_critic_B.fc2.weight.detach().numpy().copy()
      self.W_out_Q_B[:,-1] = self.main_net_critic_B.fc2.bias.detach().numpy().copy()
      
      criterion_a = np.sqrt(np.mean(self.W_out_a**2))
      criterion_Q_A = np.sqrt(np.mean(self.W_out_Q_A**2))
      criterion_Q_B = np.sqrt(np.mean(self.W_out_Q_B**2))
      if criterion_a > 1.0:
         self.beta_a = self.hyperparameters.beta_a
      else:
         self.beta_a = max(0.95*self.beta_a, self.beta_a_min)
      if max(criterion_Q_A, criterion_Q_B) > 10.0:
         self.beta_Q = self.hyperparameters.beta_Q
      else:
         self.beta_Q = max(0.95*self.beta_Q, self.beta_Q_min)


def evaluation( gtime, rds, n_ep, env_ev, ag):

   score_list = []
   for trial in range (10):
      s0, _ = env_ev.reset(seed=seed+100*(n_ep+1)+21*(trial+1))
      s0 = torch.from_numpy(s0)
      d0 = False
      trunc = False
      total = 0.0
      while not (d0 or trunc):
         if (gtime) <= rds :
            a0 = env_ev.action_space.sample()
         else:
            #a0, _ = ag.choose_action(s0, add_rand=False)
            
            with torch.no_grad():
               a0 = (ag.forward_actor( s0[None,:], add_rand=False ))[0]
               a0 = a0.detach().numpy()
            
         ns0, r0, d0, trunc, _ = env_ev.step(a0)
         ns0 = torch.from_numpy(ns0)
         total += r0
         s0 = ns0
      score_list.append(total)     
   
   return np.mean(np.array(score_list))
   

if __name__ == '__main__':
   hyper = hyperparameters()
   game = hyper.game
   env = gym.make(game)
   env_eval = gym.make(game)

   agent = Whole_Net(env,hyper)
   env.action_space.seed(seed)
   env_eval.action_space.seed(seed)
   state, info = env.reset(seed=seed)
   state = torch.from_numpy(state)

   gamma = hyper.gamma
   
   random_steps = hyper.random_steps
   start_steps = hyper.start_steps
   
   name_condition = "_game_" + game + "_gamma_" + str(gamma) + "_Nl_" + str(hyper.layer_size) + "_betaaQ_" + str(hyper.beta_a) + "_" + str(hyper.beta_Q) + "_to_" + str(hyper.beta_a_min) + "_" + str(hyper.beta_Q_min) + "_resid_" + str(hyper.resid_coef) + "_StartSteps_" + str(start_steps) + "_" + str(seed) + "th_try"

   filename_learning_curve = "LC_ourTD3" + name_condition + ".txt"
   file_learning_curve = open(filename_learning_curve, "w")
   
   filename_learning_curve_eval = "LC_eval_ourTD3" + name_condition + ".txt"
   file_learning_curve_eval = open(filename_learning_curve_eval, "w")
   
   total_reward_list = deque([], maxlen=10)
   #state = np.reshape(state[0], [1, agent.state_size])

   global_time = 0
   n_episode = 0
   while global_time < 1000000 :
      total_reward = 0.0
      total_reward_intr = 0.0
      done = False
      t = 0
      while not done:
         
         if (global_time+1) <= random_steps :
            action = env.action_space.sample()
         else:
            with torch.no_grad():
               action = (agent.forward_actor( state[None,:], add_rand=True ))[0]
               action = action.detach().numpy()

         next_state, reward, terminated, truncated, info = env.step(action)
         next_state = torch.from_numpy(next_state)
         done = terminated or truncated

         total_reward += reward
         agent.memory_append((state, action, reward, next_state, terminated))  

         t += 1
         global_time += 1

         if global_time == random_steps :
            if start_steps :
               for _ in range(random_steps):
                  #training
                  agent.TD3_train(train_actor=True, train_critic=True)
            
         if global_time > random_steps :
            #training
            agent.TD3_train(train_actor=True, train_critic=True)  

         if global_time%1000 == 0 and global_time > random_steps :
            agent.Ridge_decay()         
            
         if global_time%1000 == 0:
            if len(total_reward_list) != 0:
               mean_reward = np.mean(np.array(total_reward_list))
               last_reward = total_reward_list[-1]
               print(global_time, mean_reward, last_reward, sep="	", file=file_learning_curve)
            elif global_time==1000 and done:
               print(global_time, total_reward, total_reward, sep="	", file=file_learning_curve)
              
            if global_time%2000 == 0:
               eval_reward = evaluation( global_time, random_steps, n_episode, env_eval, agent)
               print(global_time, eval_reward, sep="	", file=file_learning_curve_eval)
            
         
         if done:
            state, info = env.reset(seed=seed+100*(n_episode+1))
            state = torch.from_numpy(state)
         else:
            state = next_state
      
      print(n_episode+1, total_reward, t, global_time, sep="	")
      total_reward_list.append(total_reward)
      n_episode += 1

   
   file_learning_curve.close()
   file_learning_curve_eval.close()

   print(time.time() - start_time)
   if hyper.time_record:
      filename_time = "time_TD3" + name_condition + ".txt"
      file_time = open(filename_time, "w")
      print(time.time() - start_time, file=file_time)
      file_time.close()
   
   if hyper.movie_record:
      #Video Recording
      import os
      from gymnasium.wrappers import RecordVideo
      currentdir = os.getcwd()
      cd_path = os.path.join(currentdir, 'movies')
      name_prefix = "movie_ourTD3" + name_condition
      env = RecordVideo( gym.make(game, render_mode='rgb_array'), video_folder=cd_path, episode_trigger = None, name_prefix=name_prefix )
      state, info = env.reset(seed=seed+10000)
      state = torch.from_numpy(state)
      total_rewards = 0.0
      done = False
      while not done:
         #action, _ = agent.choose_action(state, add_rand=False)
         with torch.no_grad():
            action = (agent.forward_actor( state[None,:], add_rand=False ))[0]
            action = action.detach().numpy()
         next_state, reward, terminated, truncated, info = env.step(action)
         next_state = torch.from_numpy(next_state)
         done = terminated or truncated
         state = next_state
      
         total_rewards += reward
      print(seed, total_rewards)
      env.close()
   
