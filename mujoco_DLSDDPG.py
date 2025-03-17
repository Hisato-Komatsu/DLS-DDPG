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
rng = np.random.default_rng(seed)
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
      self.LS_batch_size = 10000
      self.layer_size = 1024
      self.memory_size = 1000000
      self.minibatch_size = 256

      self.sigma = 0.1
      self.tau = 0.005
      self.tau_LS = 0.1
      
      self.weight_actor = 2.0    
      self.restrict_upb = 0.4 #0.4
      
      self.LS_actor = True
      self.LS_critic = True
      self.DDPG_actor = True
      self.DDPG_critic = True

      self.optim_choosing = True
      
      self.resid_coef = 0.001
      self.random_steps = 25000
      
      self.movie_record = True
      self.weight_record = True
      self.time_record = False
      
class Agent:
   def __init__(self, env, hyperparameters):
      self.env = env
      self.hyperparameters = hyperparameters
      self.state_size = env.observation_space.shape[0] 
      self.action_size = env.action_space.shape[0]
      self.lower_bound = env.action_space.low
      self.upper_bound = env.action_space.high

      self.state_list = []
      self.optim_list = []

      self.beta_Q = self.hyperparameters.beta_Q # Ridge term
      self.beta_a = self.hyperparameters.beta_a # Bayes term
      self.beta_Q_min = self.hyperparameters.beta_Q_min # Ridge term
      self.beta_a_min = self.hyperparameters.beta_a_min # Bayes term
      self.layer_size = self.hyperparameters.layer_size
      self.net = Whole_Net(env,hyperparameters=self.hyperparameters)

      self.gamma = self.hyperparameters.gamma
      self.LS_batch_size = self.hyperparameters.LS_batch_size
      self.LS_actor = self.hyperparameters.LS_actor
      self.LS_critic = self.hyperparameters.LS_critic
      
      self.weight_actor = self.hyperparameters.weight_actor
      self.restrict_upb = self.hyperparameters.restrict_upb
      
      self.optim_choosing = self.hyperparameters.optim_choosing

      self.weight_record = self.hyperparameters.weight_record
      if self.weight_record:
         self.W_in_s_actor = np.zeros( (self.layer_size,self.state_size+1) )
         self.W_in_s_critic = np.zeros( (self.layer_size,self.state_size+1) )
      self.W_in_a = np.zeros( (self.layer_size,self.action_size) )
      self.W_out_a = np.zeros( (self.action_size,self.layer_size+1) )
      self.W_out_Q = np.zeros( (1,self.layer_size+1) )
      
      self.XXT_Q = np.zeros((self.layer_size+1,self.layer_size+1)) 
      self.XXT_a = np.zeros((self.layer_size+1,self.layer_size+1)) 
      self.rX_0 = np.zeros((1, self.layer_size+1))
      self.aX_0 = np.zeros((self.action_size, self.layer_size+1))
      self.rX = np.zeros((1, self.layer_size+1))
      self.aX = np.zeros((self.action_size, self.layer_size+1))

      self.Ridge_term_a = self.beta_a*np.identity(self.layer_size+1)
      self.Ridge_term_Q = self.beta_Q*np.identity(self.layer_size+1)

      self.copy_from_torch()
      self.sigma_action = self.net.sigma
      self.tau_LS = self.hyperparameters.tau_LS


   def _ReLU(self,x):
      zeros = np.zeros(x.shape)
      return np.maximum(x,zeros)

   def copy_from_torch(self, copy_full=False):
      if copy_full:
         if self.weight_record:
            self.W_in_s_actor[:,:self.state_size] = self.net.main_net_actor.fc1.weight.detach().numpy().copy()
            self.W_in_s_actor[:,-1] = self.net.main_net_actor.fc1.bias.detach().numpy().copy()
            self.W_in_s_critic[:,:self.state_size] = self.net.main_net_critic.fc1_s.weight.detach().numpy().copy()
            self.W_in_s_critic[:,-1] = self.net.main_net_critic.fc1_s.bias.detach().numpy().copy()  
         
         self.W_out_a[:,:self.layer_size] = self.net.main_net_actor.fc2.weight.detach().numpy().copy()
         self.W_out_a[:,-1] = self.net.main_net_actor.fc2.bias.detach().numpy().copy()

      self.W_in_a = self.net.main_net_critic.fc1_a.weight.detach().numpy().copy()
      self.W_out_Q[:,:self.layer_size] = self.net.main_net_critic.fc2.weight.detach().numpy().copy()
      self.W_out_Q[:,-1] = self.net.main_net_critic.fc2.bias.detach().numpy().copy()
      
   def copy_to_torch(self, copy_actor=False, copy_critic=False):
      if copy_actor:
         self.net.main_net_actor.fc2.weight.data = torch.tensor( copy.deepcopy( self.W_out_a[:,:self.layer_size] ) ).float()
         self.net.main_net_actor.fc2.bias.data = torch.tensor( copy.deepcopy( self.W_out_a[:,-1] ) ).float()
         for target_param_a, param_a in zip(self.net.target_net_actor.parameters(), self.net.main_net_actor.parameters()):  
            target_param_a.data.copy_( target_param_a.data * (1.0 - self.tau_LS) + param_a.data * self.tau_LS )
 
      if copy_critic:
         self.net.main_net_critic.fc2.weight.data = torch.tensor( copy.deepcopy( self.W_out_Q[:,:self.layer_size] ) ).float()
         self.net.main_net_critic.fc2.bias.data = torch.tensor( copy.deepcopy( self.W_out_Q[:,-1] ) ).float()
         for target_param_c, param_c in zip(self.net.target_net_critic.parameters(), self.net.main_net_critic.parameters()):
            target_param_c.data.copy_( target_param_c.data * (1.0 - self.tau_LS) + param_c.data * self.tau_LS )

   def minusQ(self, action_0):
      X_Perc = np.concatenate( [np.tanh( self.X_in_Q + np.dot(self.W_in_a, np.array(action_0)) ), [1.0]] )
      return -np.dot(self.W_out_Q, X_Perc).item()

   def minus_diffQ(self, action_0):
      diff_X_Perc = 1.0 - np.tanh( self.X_in_Q + np.dot(self.W_in_a, np.array(action_0)) )**2
      WdfX = self.W_out_Q[:,:-1]*diff_X_Perc.reshape(1,-1)
      return -np.dot(WdfX, self.W_in_a).reshape(-1)

   def choose_action(self, state, add_rand=True):
      with torch.no_grad():
         policy_clip = self.net.forward_actor( state[None,:] , add_rand=False, return_hidden=False, return_res=False)
      policy_clip = policy_clip.squeeze(0).detach().numpy()
      
      if self.LS_actor or self.optim_choosing :
         with torch.no_grad():
            self.X_in_Q = self.net.forward_critic_midstate(state[None,:])
            self.X_in_Q = self.X_in_Q.squeeze(0).detach().numpy() 
         bounds = np.array( [policy_clip-self.restrict_upb,policy_clip+self.restrict_upb] )
         bounds = np.clip( bounds, self.lower_bound, self.upper_bound )
         bounds = list( bounds.T )
         optim, _, _ = fmin_l_bfgs_b(self.minusQ,x0=policy_clip,fprime=self.minus_diffQ, bounds=bounds, maxiter=10, disp=None )
      else:
         optim = policy_clip
      
      if add_rand:
         #rand_exp = np.random.normal(loc=0.0,scale=self.sigma_action, size=policy_clip.shape)
         rand_exp = rng.normal(loc=0.0, scale=self.sigma_action, size=policy_clip.shape)
      else:
         rand_exp = 0.0
   
      if self.optim_choosing:
         action_chosen = np.clip(optim+rand_exp, self.lower_bound, self.upper_bound)
      else:
         action_chosen = np.clip(policy_clip+rand_exp, self.lower_bound, self.upper_bound)
      action_chosen = np.float32(action_chosen)

      return action_chosen, optim # returns action and corresponding X_res

   def calculate_Wout(self, train_actor=False, train_critic=False): 
      #print( np.mean(self.W_out_a**2), np.mean(self.W_out_Q**2), np.mean(self.W_in_s_actor**2), np.mean(self.W_in_s_critic**2) )
      self.Ridge_decay()
      sample_size = self.LS_batch_size
      weight_a = self.weight_actor 
      if train_actor or train_critic:
         self.LS_replay(replay_actor=train_actor, replay_critic=train_critic, weight_a=weight_a, sample_size=sample_size)

      if train_critic:   

         XXTinv_Q = np.linalg.inv( self.XXT_Q_0 + (sample_size)*self.Ridge_term_Q )
         self.W_out_Q = np.dot(self.rX_0, XXTinv_Q)
         
         #print( np.sqrt(np.mean(self.XXT_Q_0**2)), np.sqrt(np.mean(self.rX_0**2)) )
         #print( np.sqrt(np.mean(self.XXT_Q**2)), np.sqrt(np.mean(self.rX**2)) )

      if train_actor:
         with torch.no_grad():
            _, _, Xi_a_temp = self.net.forward_actor( torch.stack(self.state_list), add_rand=False, return_hidden=True, return_res=False)
         Xi_a_temp_array_0 = Xi_a_temp.detach().numpy()
         Xi_a_temp_array = np.concatenate( [Xi_a_temp_array_0, np.ones((Xi_a_temp_array_0.shape[0],1))], axis=1)
         self.XXT_a = copy.deepcopy(self.XXT_a_0) + np.dot(Xi_a_temp_array.T , Xi_a_temp_array )
         self.aX = copy.deepcopy(self.aX_0) + np.dot(np.array(self.optim_list).reshape(-1,self.action_size).T, Xi_a_temp_array )
         XXTinv_a = np.linalg.inv( self.XXT_a )
         self.W_out_a = np.dot(self.aX, XXTinv_a)

         #print( np.sqrt(np.mean(self.XXT_a_0**2)), np.sqrt(np.mean(self.aX_0**2)) )
         #print( np.sqrt(np.mean(self.XXT_a**2)), np.sqrt(np.mean(self.aX**2)) )

      #print( np.mean(self.W_out_a**2), np.mean(self.W_out_Q**2), np.mean(self.W_in_s_actor**2), np.mean(self.W_in_s_critic**2) )

      self.state_list = []
      self.optim_list = []

   def Ridge_decay(self):
      criterion_a = np.sqrt(np.mean(self.W_out_a**2))
      criterion_Q = np.sqrt(np.mean(self.W_out_Q**2))
      if criterion_a > 1.0:
         self.beta_a = self.hyperparameters.beta_a
      else:
         self.beta_a = max(0.95*self.beta_a, self.beta_a_min)
      if criterion_Q > 10.0:
         self.beta_Q = self.hyperparameters.beta_Q
      else:
         self.beta_Q = max(0.95*self.beta_Q, self.beta_Q_min)
      self.Ridge_term_a = self.beta_a*np.identity(self.layer_size+1)
      self.Ridge_term_Q = self.beta_Q*np.identity(self.layer_size+1)
      self.net.beta_a = self.beta_a
      self.net.beta_Q = self.beta_Q

   def LS_replay(self, replay_actor, replay_critic, weight_a, sample_size):
      s_sample, a_sample, r_sample, next_s_sample, done_sample = self.net.memory_sample(sample_size = sample_size)
      
      if replay_critic:
         with torch.no_grad():
            Xi_0 = self.net.forward_critic_hidden(s_sample, a_sample)
            Q_ns = self.net.target_q( next_s_sample )

         Xi_p0_array_0 = Xi_0.detach().numpy()
         Xi_p0_array = np.concatenate( [Xi_p0_array_0, np.ones((Xi_p0_array_0.shape[0],1))], axis=1)

         r_array = r_sample.detach().numpy().reshape(1,-1)
         Q_ns_array = Q_ns.detach().numpy().reshape(1,-1)
         done_array = done_sample.float().detach().numpy().reshape(1,-1)
         targ_array = r_array + self.gamma*(1.0-done_array)*Q_ns_array

         self.XXT_Q_0 = np.dot(Xi_p0_array.T , Xi_p0_array )
         self.rX_0 = np.dot(targ_array, Xi_p0_array )
      
      if replay_actor:
         with torch.no_grad():
            ac, _, Xi_a = self.net.forward_actor(s_sample, add_rand=False, return_hidden=True, return_res=False)

         Xi_a_array_0 = Xi_a.detach().numpy()
         Xi_a_array = np.concatenate( [Xi_a_array_0, np.ones((Xi_a_array_0.shape[0],1))], axis=1)
         #ac_array = ac.detach().numpy()
         
         self.XXT_a_0 = weight_a*np.dot(Xi_a_array.T , Xi_a_array ) + (weight_a*sample_size)*self.Ridge_term_a
         #self.aX_0 = weight_a*np.dot(ac_array.T, Xi_a_array )
         self.aX_0 = np.dot(self.W_out_a, self.XXT_a_0 )


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
      self.beta_Q = self.hyperparameters.beta_Q # Ridge term
      self.beta_a = self.hyperparameters.beta_a # Bayes term

      self.layer_size = self.hyperparameters.layer_size
      self.sigma = self.hyperparameters.sigma
      self.tau = self.hyperparameters.tau

      #self.activation = 'tanh' 

      self.main_net_actor = Actor_Net(env, layer_size=self.layer_size)
      self.main_net_critic = Critic_Net(env, layer_size=self.layer_size)
      self.target_net_actor = copy.deepcopy(self.main_net_actor)
      self.target_net_critic = copy.deepcopy(self.main_net_critic)

      self.memory_size = self.hyperparameters.memory_size
      self.minibatch_size = self.hyperparameters.minibatch_size
      self.memory = []
      self.memory_full = False

      self.optimizer_actor = torch.optim.Adam(list( self.main_net_actor.parameters() ), lr=hyper.learning_rate)
      self.optimizer_critic = torch.optim.Adam(list( self.main_net_critic.parameters() ), lr=hyper.learning_rate)
      
      self.resid_coef = self.hyperparameters.resid_coef 

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
      if return_hidden:
         if return_res:
            resid = torch.mean( (mu - mu_clip)**2 )
            return mu_clip, torch.clip( mu, self.lower_bound, self.upper_bound), u, resid
         else:
            return mu_clip, torch.clip( mu, self.lower_bound, self.upper_bound), u
      else:
         if return_res:
            resid = torch.mean( (mu - mu_clip)**2 )
            return mu_clip, resid
         else:
            return mu_clip

   def forward_critic_midstate(self, x):
      x = x.float()
      us = torch.flatten(x, 1)
      
      um = self.main_net_critic.fc1_s(us)
      return um

   def forward_critic_hidden(self, x, a):
      x = x.float()
      a = a.float()
      us = torch.flatten(x, 1)
      ua = torch.flatten(a, 1)
      
      u = torF.tanh(self.main_net_critic.fc1_s(us) + self.main_net_critic.fc1_a(ua))
      return u

   def forward_critic(self, x, a):
      u = self.forward_critic_hidden(x, a)
      q = self.main_net_critic.fc2(u).squeeze(1)
      return q

   def target_q(self, x):
      x = x.float()
      ut = torch.flatten(x, 1) # flatten all dimensions except minibatch
      
      ut = torF.tanh(self.target_net_actor.fc1(ut))
      mut = self.target_net_actor.fc2(ut)
      mut = torch.clip( mut, self.lower_bound, self.upper_bound)

      vt = torch.flatten(x, 1)
      
      vt = torF.tanh(self.target_net_critic.fc1_s(vt) + self.target_net_critic.fc1_a(mut))
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
         loss_critic = loss_critic + self.beta_Q*( torch.sum(self.main_net_critic.fc1_s.weight**2) + torch.sum(self.main_net_critic.fc1_s.bias**2) + torch.sum(self.main_net_critic.fc1_a.weight**2) + torch.sum(self.main_net_critic.fc2.weight**2) + torch.sum(self.main_net_critic.fc2.bias**2) ) 
         
         self.optimizer_critic.zero_grad()
         loss_critic.backward()
         #nn.utils.clip_grad_norm_(self.main_net_critic.parameters(), max_grad_norm)
         self.optimizer_critic.step()
         
         del loss_critic

      if train_actor:
         mu_actor, resid = self.forward_actor( s_sample, add_rand=False, return_hidden=False, return_res=True )
         main_q_actor = self.forward_critic( s_sample, mu_actor )
         loss_actor = - torch.mean( main_q_actor ) 
         loss_actor = loss_actor + self.beta_a*( torch.sum(self.main_net_actor.fc1.weight**2) + torch.sum(self.main_net_actor.fc1.bias**2) + torch.sum(self.main_net_actor.fc2.weight**2) + torch.sum(self.main_net_actor.fc2.bias**2) ) 
         loss_actor = loss_actor + self.resid_coef*resid
         
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
   if (gtime+1) > rds : #12/25 (gtime+1) -> gtime ? ... maybe no need to change
      ag.copy_from_torch()

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
            a0, _ = ag.choose_action(s0, add_rand=False)
            """
            with torch.no_grad():
               a0 = (ag.net.forward_actor( torch.from_numpy(s0)[None,:], add_rand=False ))[0]
               a0 = a0.detach().numpy()
            """
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

   agent = Agent(env,hyper)
   env.action_space.seed(seed)
   env_eval.action_space.seed(seed)
   state, info = env.reset(seed=seed)
   state = torch.from_numpy(state)

   #max_grad_norm = hyper.max_grad_norm
   LS_a = hyper.LS_actor
   LS_c = hyper.LS_critic
   DDPG_a = hyper.DDPG_actor
   DDPG_c = hyper.DDPG_critic
   
   random_steps = hyper.random_steps
   
   name_condition = "_LS_" + str(int(LS_a)) + str(int(LS_c)) + "_DDPG_" + str(int(DDPG_a)) + str(int(DDPG_c)) + "_BOAC_" + str(int(hyper.optim_choosing)) + "_game_" + game + "_gamma_" + str(hyper.gamma) + "_Nl_" + str(hyper.layer_size) + "_betaaQ_" + str(hyper.beta_a) + "_" + str(hyper.beta_Q) + "_to_" + str(hyper.beta_a_min) + "_" + str(hyper.beta_Q_min) + "_weighta_" + str(hyper.weight_actor) + "_upb_" + str(hyper.restrict_upb) + "_resid_" + str(hyper.resid_coef) + "_" + str(seed) + "th_try"

   filename_learning_curve = "LC_DLSDDPG_softT_offC" + name_condition + ".txt"
   file_learning_curve = open(filename_learning_curve, "w")
   
   filename_learning_curve_eval = "LC_eval_DLSDDPG_softT_offC" + name_condition + ".txt"
   file_learning_curve_eval = open(filename_learning_curve_eval, "w")
   
   weight_record = hyper.weight_record
   if weight_record:
      filename_weight = "weight_DLSDDPG_softT_offC" + name_condition + ".txt"
      file_weight = open(filename_weight, "w")
   
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
            agent.copy_from_torch()
            action, optim = agent.choose_action(state, add_rand=True)

         next_state, reward, terminated, truncated, info = env.step(action)
         next_state = torch.from_numpy(next_state)
         done = terminated or truncated

         total_reward += reward
         agent.net.memory_append((state, action, reward, next_state, terminated))  

         t += 1
         global_time += 1

         if global_time == random_steps :
            for _ in range(random_steps):
               #training
               agent.net.DDPG_train(train_actor=True, train_critic=True)
            
         if global_time > random_steps :
            #training
            agent.net.DDPG_train(train_actor=DDPG_a, train_critic=DDPG_c )
            agent.state_list.append(state)
            agent.optim_list.append(optim)
         
         if global_time%1000 == 0 and global_time > random_steps :
            agent.copy_from_torch(copy_full=True)
            agent.calculate_Wout(train_actor=LS_a, train_critic=LS_c)
            agent.copy_to_torch(copy_actor=LS_a, copy_critic=LS_c)
            
         if global_time%1000 == 0:
            if len(total_reward_list) != 0:
               mean_reward = np.mean(np.array(total_reward_list))
               last_reward = total_reward_list[-1]
               print(global_time, mean_reward, last_reward, sep="	", file=file_learning_curve)
            elif global_time==1000 and done:
               print(global_time, total_reward, total_reward, sep="	", file=file_learning_curve)
              
            if weight_record:
               agent.copy_from_torch(copy_full=True)
               print(global_time, np.mean(agent.W_out_a**2), np.mean(agent.W_out_Q**2), np.mean(agent.W_in_s_actor**2), np.mean(agent.W_in_s_critic**2), sep="	", file=file_weight)
             
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
   if weight_record:
      file_weight.close()

   print(time.time() - start_time)
   if hyper.time_record:
      filename_time = "time_DLSDDPG_softT_offC" + name_condition + ".txt"
      file_time = open(filename_time, "w")
      print(time.time() - start_time, file=file_time)
      file_time.close()
   
   if hyper.movie_record:
      #Video Recording
      #import os
      from gymnasium.wrappers import RecordVideo
      currentdir = os.getcwd()
      cd_path = os.path.join(currentdir, 'movies')
      name_prefix = "movie" + name_condition
      env = RecordVideo( gym.make(game, render_mode='rgb_array'), video_folder=cd_path, episode_trigger = None, name_prefix=name_prefix )
      state, info = env.reset(seed=seed+10000)
      state = torch.from_numpy(state)
      total_rewards = 0.0
      done = False
      agent.copy_from_torch()
      while not done:
         action, _ = agent.choose_action(state, add_rand=False)
         next_state, reward, terminated, truncated, info = env.step(action)
         next_state = torch.from_numpy(next_state)
         done = terminated or truncated
         state = next_state
      
         total_rewards += reward
      print(seed, LS_a, LS_c, total_rewards)
      env.close()
   
