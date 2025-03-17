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
      #self.beta_Q = 0.0
      #self.beta_a = 0.0
      #self.beta_Q_min = 1.0e-3
      #self.beta_a_min = 1.0e-3

      self.learning_rate = 1.0e-3
      #self.LS_batch_size = 10000
      self.layer_size1 = 400
      self.layer_size2 = 300
      self.memory_size = 1000000
      self.minibatch_size = 256

      self.sigma = 0.1
      self.tau = 0.005
      #self.tau_LS = 0.1
      
      """
      self.weight_actor = 2.0
      self.weight_critic = 2.0      
      self.restrict_upb = 0.4
      
      self.LS_actor = True
      self.LS_critic = True
      self.DDPG_actor = True
      self.DDPG_critic = True

      self.optim_choosing = True
      
      self.resid_coef = 0.001
      """
      self.random_steps = 10000
      
      self.movie_record = True
      #self.weight_record = False
      self.time_record = False


class Actor_Net(nn.Module):
   def __init__(self, env, layer_size1, layer_size2):
      super().__init__()
      self.state_size = env.observation_space.shape[0] 
      self.action_size = env.action_space.shape[0]
      self.layer_size1 = layer_size1
      self.layer_size2 = layer_size2

      self.fc0 = nn.Linear(self.state_size, self.layer_size1)
      self.fc1 = nn.Linear(self.layer_size1, self.layer_size2)
      self.fc2 = nn.Linear(self.layer_size2, self.action_size)

class Critic_Net(nn.Module):
   def __init__(self, env, layer_size1, layer_size2):
      super().__init__()
      self.state_size = env.observation_space.shape[0] 
      self.action_size = env.action_space.shape[0]
      self.layer_size1 = layer_size1
      self.layer_size2 = layer_size2

      self.fc0 = nn.Linear(self.state_size+self.action_size, self.layer_size1)
      self.fc1 = nn.Linear(self.layer_size1, self.layer_size2)
      self.fc2 = nn.Linear(self.layer_size2, 1)

class Whole_Net():
   def __init__(self, env, hyperparameters):
      super().__init__()
      self.hyperparameters = hyperparameters
      self.state_size = env.observation_space.shape[0] 
      self.action_size = env.action_space.shape[0]
      self.lower_bound = torch.tensor(env.action_space.low)
      self.upper_bound = torch.tensor(env.action_space.high)

      self.gamma = self.hyperparameters.gamma
      #self.beta_Q = self.hyperparameters.beta_Q # Ridge term
      #self.beta_a = self.hyperparameters.beta_a # Bayes term

      self.layer_size1 = self.hyperparameters.layer_size1
      self.layer_size2 = self.hyperparameters.layer_size2
      self.sigma = self.hyperparameters.sigma
      self.tau = self.hyperparameters.tau

      self.main_net_actor = Actor_Net(env, layer_size1=self.layer_size1, layer_size2=self.layer_size2)
      self.main_net_critic = Critic_Net(env, layer_size1=self.layer_size1, layer_size2=self.layer_size2)
      self.target_net_actor = copy.deepcopy(self.main_net_actor)
      self.target_net_critic = copy.deepcopy(self.main_net_critic)

      self.memory_size = self.hyperparameters.memory_size
      self.minibatch_size = self.hyperparameters.minibatch_size
      self.memory = []
      self.memory_full = False

      self.optimizer_actor = torch.optim.Adam(list( self.main_net_actor.parameters() ), lr=hyper.learning_rate)
      self.optimizer_critic = torch.optim.Adam(list( self.main_net_critic.parameters() ), lr=hyper.learning_rate)
      
      #self.resid_coef = self.hyperparameters.resid_coef 

   def forward_actor(self, x, add_rand=True):
      x = x.float()
      u = torch.flatten(x, 1) # flatten all dimensions except minibatch

      u = torF.relu(self.main_net_actor.fc0(u))
      u = torF.relu(self.main_net_actor.fc1(u))
      mu = self.upper_bound*torF.tanh(self.main_net_actor.fc2(u))
      if add_rand:
         mu_rand = torch.normal(mean=torch.zeros(mu.shape), std=self.sigma*torch.ones(mu.shape))
         mu_clip = torch.clip( mu + mu_rand, self.lower_bound, self.upper_bound )
      else:
         mu_clip = mu 
      return mu_clip

   def forward_critic(self, x, a):
      x = x.float()
      a = a.float()
      us = torch.flatten(x, 1)
      ua = torch.flatten(a, 1)
      
      u = torch.cat([us,ua], dim=1)
      u = torF.relu(self.main_net_critic.fc0(u))
      u = torF.relu(self.main_net_critic.fc1(u))
      q = self.main_net_critic.fc2(u).squeeze(1)
      return q


   def target_q(self, x):
      x = x.float()
      ut0 = torch.flatten(x, 1) # flatten all dimensions except minibatch
      
      ut = torF.relu(self.target_net_actor.fc0(ut0))
      ut = torF.relu(self.target_net_actor.fc1(ut))
      mut = self.upper_bound*torF.tanh(self.target_net_actor.fc2(ut))

      vt = torch.cat([ut0,mut], dim=1)
      
      vt = torF.relu(self.target_net_critic.fc0(vt))
      vt = torF.relu(self.target_net_critic.fc1(vt))
      qt = self.target_net_critic.fc2(vt).squeeze(1)
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

   def DDPG_train(self, train_actor=False, train_critic=False):

      s_sample, a_sample, r_sample, next_s_sample, done_sample = self.memory_sample(sample_size = self.minibatch_size)
      if train_critic:
         with torch.no_grad():
            target_q = self.target_q( next_s_sample )
         target = r_sample + self.gamma*target_q*(1 - done_sample.float())
         main_q_critic = self.forward_critic( s_sample, a_sample )
         loss_critic = torF.mse_loss(main_q_critic.float(), target.float(), reduction='mean') 
         
         self.optimizer_critic.zero_grad()
         loss_critic.backward()
         #nn.utils.clip_grad_norm_(self.main_net_critic.parameters(), max_grad_norm)
         self.optimizer_critic.step()
         
         del loss_critic

      if train_actor:
         mu_actor = self.forward_actor( s_sample, add_rand=False )
         main_q_actor = self.forward_critic( s_sample, mu_actor )
         loss_actor = - torch.mean( main_q_actor ) 
         
         self.optimizer_actor.zero_grad()
         loss_actor.backward()
         #nn.utils.clip_grad_norm_(self.main_net_actor.parameters(), max_grad_norm)
         self.optimizer_actor.step()
         
         del loss_actor

      self.update_target(update_actor=train_actor, update_critic=train_critic)

   def update_target(self, update_actor=False, update_critic=False):
      if update_actor:
         for target_param_a, param_a in zip(self.target_net_actor.parameters(), self.main_net_actor.parameters()):
            target_param_a.data.copy_( target_param_a.data * (1.0 - self.tau) + param_a.data * self.tau )
      if update_critic:
         for target_param_c, param_c in zip(self.target_net_critic.parameters(), self.main_net_critic.parameters()):
            target_param_c.data.copy_( target_param_c.data * (1.0 - self.tau) + param_c.data * self.tau )



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
   """
   max_grad_norm = hyper.max_grad_norm
   LS_a = hyper.LS_actor
   LS_c = hyper.LS_critic
   DDPG_a = hyper.DDPG_actor
   DDPG_c = hyper.DDPG_critic
   """
   
   random_steps = hyper.random_steps
   
   name_condition = "_game_" + game + "_gamma_" + str(gamma) + "_Nl_" + str(hyper.layer_size1) + "_" + str(hyper.layer_size2) + "_" + str(seed) + "th_try"

   filename_learning_curve = "LC_DDPG" + name_condition + ".txt"
   file_learning_curve = open(filename_learning_curve, "w")
   
   filename_learning_curve_eval = "LC_eval_DDPG" + name_condition + ".txt"
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
            for _ in range(random_steps):
               #training
               agent.DDPG_train(train_actor=True, train_critic=True)
            
         if global_time > random_steps :
            #training
            agent.DDPG_train(train_actor=True, train_critic=True)  
         
            
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
      filename_time = "time_DDPG" + name_condition + ".txt"
      file_time = open(filename_time, "w")
      print(time.time() - start_time, file=file_time)
      file_time.close()
   
   if hyper.movie_record:
      #Video Recording
      import os
      from gymnasium.wrappers import RecordVideo
      currentdir = os.getcwd()
      cd_path = os.path.join(currentdir, 'movies')
      name_prefix = "movie" + name_condition
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
   
