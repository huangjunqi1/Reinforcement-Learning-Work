import gym
import numpy as np
import tqdm
import random
import math
from typing import Tuple, Dict, Any
import os
class DiscreteCartPoleEnv(gym.Env):
    def __init__(self, intervals=16):
        self._env = gym.make('CartPole-v1')
        self.action_space = self._env.action_space
        self.intervals = intervals
        self.observation_space = gym.spaces.MultiDiscrete([intervals]*4)
        self._to_discrete = lambda x, a, b: int(min(max(0, (x-a)*self.intervals/(b-a)), self.intervals))
        
    def render(self):
        self._env.render()
    
    def reset(self):
        state, _ = self._env.reset()
        return self._discretize(state)

    def _discretize(self, state:np.array)->Tuple:
        cart_pos, cart_v, pole_angle, pole_v = state
        cart_pos = self._to_discrete(cart_pos, -2.4, 2.4)
        cart_v = self._to_discrete(cart_v, -3.0, 3.0)
        pole_angle = self._to_discrete(pole_angle, -0.5, 0.5)
        pole_v = self._to_discrete(pole_v, -2.0, 2.0)
        return (cart_pos, cart_v, pole_angle, pole_v)
    
    def step(self, action:int)->Tuple[Tuple, float, bool, Any]:
        state, reward, done, _ , info = self._env.step(action)
        state = self._discretize(state)
        return state, reward, done, info

class QLearner:
    def __init__(self, config:Dict):
        for k, v in config.items():
            setattr(self, k, v)
        self.epsilon = self.epsilon_lower
        self.lr = self.lr_upper
        self.buffer = list()
        self.buffer_pointer = 0

    
    def add_to_buffer(self, data):
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(data)
        else:
            self.buffer[self.buffer_pointer] = data
        self.buffer_pointer += 1
        self.buffer_pointer %= self.buffer_size
    
    def sample_batch(self):
        return random.sample(self.buffer, self.batch_size)
    
    def greedy(self, state:Tuple)->int:
        return self.q[state].argmax()

    def epsilon_greedy(self, state:Tuple)->int:
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        return self.greedy(state)
    
    def epsilon_decay(self, total_step):
        self.epsilon = self.epsilon_lower + (self.epsilon_upper - self.epsilon_lower) * math.exp(-total_step / self.epsilon_decay_freq)
    
    def lr_decay(self, total_step):
        self.lr = self.lr_lower + (self.lr_upper - self.lr_lower) * math.exp(-total_step / self.lr_decay_freq)
    
    def update_q(self, total_step):
        if total_step % self.update_freq != 0 or len(self.buffer) < self.batch_size:
            return
        batch = self.sample_batch()
        for state, action, reward, new_state in batch:
            self.q[state][action] += self.lr * (self.gamma * self.q[new_state].max() + reward - self.q[state][action])
    
    def train(self):
        total_step = 0
        for i in tqdm.trange(self.start_iter, self.iter):
            state = self.env.reset()
            done = False
            while not done:
                total_step += 1
                action = self.epsilon_greedy(state)
                self.epsilon_decay(total_step)
                new_state, reward, done, _ = self.env.step(action)
                if self.render:
                    self.env.render()
                if done:
                    reward = self.end_reward
                self.add_to_buffer((state, action, reward, new_state))
                self.q[state][action] += self.lr * (self.gamma * self.q[new_state].max() + reward - self.q[state][action])
                #self.update_q(total_step)
                self.lr_decay(total_step)
                
                state = new_state
            self.save_model(i)
    def save_model(self, i):
        if i % self.save_freq == 0:
            np.save(os.path.join(self.save_path, f'{i}.npy'), self.q)
        
class TDnLearner:
    def __init__(self, config:Dict):
        for k, v in config.items():
            setattr(self, k, v)
        self.epsilon = self.epsilon_lower
        self.lr = self.lr_upper
        self.buffer = list()
        self.buffer_pointer = 0
    
    def add_to_buffer(self, data):
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(data)
        else:
            self.buffer[self.buffer_pointer] = data
        self.buffer_pointer += 1
        self.buffer_pointer %= self.buffer_size
    
    def sample_batch(self):
        return random.sample(self.buffer, self.batch_size)
    
    def greedy(self, state:Tuple)->int:
        return self.q[state].argmax()

    def epsilon_greedy(self, state:Tuple)->int:
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        return self.greedy(state)
    
    def epsilon_decay(self, total_step):
        self.epsilon = self.epsilon_lower + (self.epsilon_upper - self.epsilon_lower) * math.exp(-total_step / self.epsilon_decay_freq)
    
    def lr_decay(self, total_step):
        self.lr = self.lr_lower + (self.lr_upper - self.lr_lower) * math.exp(-total_step / self.lr_decay_freq)
    
    def update_q(self, total_step):
        if total_step % self.update_freq != 0 or len(self.buffer) < self.batch_size:
            return
        batch = self.sample_batch()
        for state, action, reward, new_state in batch:
            self.q[state][action] += self.lr * (self.gamma * self.q[new_state].max() + reward - self.q[state][action])
    
    def train(self):
        total_step = 0
        for i in tqdm.trange(self.start_iter, self.iter):
            state = self.env.reset()
            done = False
            t = 0
            T = -100
            rewards = [0]
            states = [state]
            actions = []
            while True:
                if (T == -100 or t < T):
                    total_step += 1
                    if (t == 0):
                        action = self.epsilon_greedy(state)
                        actions.append(action)
                    else:
                        action = actions[t]
                    self.epsilon_decay(total_step)
                    new_state, reward, done, _ = self.env.step(action)
                    if self.render:
                        self.env.render()
                    if done:
                        reward = self.end_reward
                        T = t + 1
                    else:
                        new_action = self.epsilon_greedy(new_state)
                        actions.append(new_action)
                    rewards.append(reward)
                    states.append(new_state)
                tau:int = t - self.TDn + 1
                if (tau >= 0):
                    G = 0.0
                    loop_upper = 0
                    if (T == -100): loop_upper = tau + self.TDn
                    else: loop_upper = min(tau+self.TDn,T)
                    for j in range(tau + 1, loop_upper + 1):
                        G += pow(self.gamma,j-tau-1)*rewards[j]
                    if (T == -100 or tau + self.TDn < T):
                        G += pow(self.gamma,self.TDn) * self.q[states[tau+self.TDn]][actions[tau+self.TDn]]
                    self.q[states[tau]][actions[tau]] += self.lr*(G-self.q[states[tau]][actions[tau]])
                
                if (T == -100 or t < T):
                    self.add_to_buffer((state, action, reward, new_state))
                    #self.update_q(total_step)
                    self.lr_decay(total_step)
                    
                    state = new_state
                t += 1
                if (tau == T-1): break
            self.save_model(i)
    def save_model(self, i):
        if i % self.save_freq == 0:
            np.save(os.path.join(self.save_path, f'{i}.npy'), self.q)            
             
if __name__ == '__main__':
    env_name = 'DiscreteCartPole'
    save_path = '5_TD_1_q_tables'
    intervals = 8
    
    env = DiscreteCartPoleEnv(intervals)
    
    q_table = np.zeros(shape=(intervals+1,)*env.observation_space.shape[0]+(env.action_space.n,))
    
    latest_checkpoint = 0
    
    if save_path not in os.listdir():
        os.mkdir(save_path)
    elif len(os.listdir(save_path)) != 0:
        latest_checkpoint = max([int(file_name.split('.')[0]) for file_name in os.listdir(save_path)])
        print(f'{latest_checkpoint}.npy loaded')
        q_table = np.load(os.path.join(save_path, f'{latest_checkpoint}.npy'))
            
    trainer = TDnLearner({
        'env':env,
        'env_name':env_name,
        'render':False,
        'end_reward':-1,
        'q':q_table,
        'start_iter':latest_checkpoint,
        'iter':latest_checkpoint+200,
        'batch_size':1,
        'buffer_size':20000,
        'gamma':0.9,
        'update_freq':1,
        'epsilon_lower':0.05,
        'epsilon_upper':0.8,
        'epsilon_decay_freq':200,
        'lr_lower':0.05,
        'lr_upper':0.5,
        'lr_decay_freq':200,
        'save_path':save_path,
        'save_freq':1,
        'TDn': 5
    })
    trainer.train()
    
    state = env.reset()
    done = False
    episode_reward = 0
    while not done:
        action = trainer.greedy(state)
        state, reward, done, _ = env.step(action)
        episode_reward += reward
        #env.render()
    print(episode_reward)
    env.close()