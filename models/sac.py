import numpy as np
import random
import torch
import torch.nn as nn
from copy import deepcopy
from collections import deque

from models.linear_transformations import transform_interval
from models.q_models import MuModel


class SAC:

    def __init__(self, action_min, action_max, q_model1, q_model2, pi_model, noise,
                 lr=1e-3, gamma=1, batch_size=128, tau=1e-2, memory_len=6000000):

        self.action_min = torch.FloatTensor(action_min)
        self.action_max = torch.FloatTensor(action_max)

        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.memory = deque(maxlen=memory_len)
        self.noise = noise


        self.q_model1 = q_model1
        self.q_target_model1 = deepcopy(self.q_model1)
        self.q_model2 = q_model2
        self.q_target_model2 = deepcopy(self.q_model2)
        self.pi_model = pi_model

        
        self.q_optimizer = torch.optim.Adam(list(self.q_model1.parameters()) + list(self.q_model2.parameters()), lr=self.lr)
        self.pi_optimizer = torch.optim.Adam(list(self.pi_model.parameters()), lr=self.lr)
        
        self.log_alpha = torch.zeros(1, requires_grad=True, dtype=torch.float32)
        self.alpha = self.log_alpha.exp()
        self.target_entropy = -np.prod(action_min.shape).item()
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.lr)

            

    def get_action(self, state):
        state = torch.FloatTensor(state)
        action, _, _ = self.pi_model(state)
        action = transform_interval(action, self.action_min, self.action_max)
        return np.clip(action.detach().numpy(), self.action_min.numpy(), self.action_max.numpy())

    def update_target_model(self, target_model, model, optimizer, loss):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        for target_param, param in zip(target_model.parameters(), model.parameters()):
            target_param.data.copy_((1 - self.tau) * target_param.data + self.tau * param.data)
        return None

    def fit(self, step):
        self.memory.append(step)

        if len(self.memory) >= self.batch_size:
            
            batch = random.sample(self.memory, self.batch_size)
            states, actions, rewards, dones, next_states = map(torch.FloatTensor, zip(*batch))
            
            rewards = rewards.reshape(self.batch_size, 1)
            dones = dones.reshape(self.batch_size, 1)
            
            with torch.no_grad():         
                nxt_action, nxt_log_prob, nxt_mean = self.pi_model(next_states)
                pred_next_actions = transform_interval(nxt_action, self.action_min, self.action_max)
                next_states_and_pred_next_actions = torch.cat((next_states, pred_next_actions), dim=1)
                q_target1 = self.q_target_model1(next_states_and_pred_next_actions)
                q_target2 = self.q_target_model2(next_states_and_pred_next_actions)
            
            targets = rewards + (1 - dones) * self.gamma * ( torch.min(q_target1, q_target2) - (self.alpha*nxt_log_prob))
            states_and_actions = torch.cat((states, actions), dim=1)
            
            
            pred_Q1 = self.q_model1(states_and_actions) 
            pred_Q2 = self.q_model2(states_and_actions) 
            q_loss1 = torch.mean((pred_Q1 - targets.detach()) ** 2)
            q_loss2 = torch.mean((pred_Q2 - targets.detach()) ** 2)
            
            q_loss = q_loss1 + q_loss2
        
            self.q_optimizer.zero_grad()
            q_loss.backward()
            self.q_optimizer.step()
        

            # actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

            pre_action, pre_log_prob, pre_mean = self.pi_model(states)
            pred_actions = transform_interval(pre_action, self.action_min, self.action_max)
            states_and_pred_actions = torch.cat((states, pred_actions), dim=1)
            
            
            
            pi_loss = torch.mean((self.alpha * pre_log_prob) - torch.min(self.q_model1(states_and_pred_actions), self.q_model2(states_and_pred_actions)))
            self.pi_optimizer.zero_grad()
            pi_loss.backward()
            self.pi_optimizer.step()
            

            alpha_loss = -(self.log_alpha * (pre_log_prob + self.target_entropy).detach()).mean()
                        
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()
        
        
        for target_param, param in zip(self.q_target_model1.parameters(), self.q_model1.parameters()):
            target_param.data.copy_((1 - self.tau) * target_param.data + self.tau * param.data)
        for target_param, param in zip(self.q_target_model2.parameters(), self.q_model2.parameters()):
            target_param.data.copy_((1 - self.tau) * target_param.data + self.tau * param.data)

        return None
    
    def save(self, path):
        torch.save({
            'q-model': self.q_model.state_dict(),
            'pi-model': self.pi_model.state_dict(),
            'noise': self.noise.state_dict(),
            'action_min': self.action_min,
            'action_max': self.action_max,
            'tau': self.tau,
            'lr': self.lr,
            'gamma': self.gamma,
            'batch_size': self.batch_size
        }, path)