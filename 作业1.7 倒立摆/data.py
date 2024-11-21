import numpy as np
import gym
import tqdm
import random
from typing import Tuple, Dict, Any
import os
import json
import matplotlib.pyplot as plt

class DiscreteCartPoleEnv(gym.Env):
    def __init__(self, intervals):
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

class Trainer:
    def __init__(self, config:Dict):
        for k, v in config.items():
            setattr(self, k, v)
        self.epsilon = self.epsilon_lower
        self.lr = self.lr_upper
        self.buffer = list()
        self.buffer_pointer = 0

    def greedy(self, state:Tuple)->int:
        return self.q[state].argmax()

intervals = 8

indexes = ["QLearning","1_TD","3_TD","5_TD","10_TD"]
env = DiscreteCartPoleEnv(intervals)
fig,ax = plt.subplots()

ans_ = []

for index in indexes:
    input_path = index + "_1_q_tables"
    ans = []
    preans_ = []
    preans = 0
    for file_name in os.listdir(input_path):
        checkpoint = int(file_name.split('.')[0])
        q_table = np.load(os.path.join(input_path, f'{checkpoint}.npy'))
        
        trainer = Trainer({
            'env':env,
            'render':False,
            'end_reward':-1,
            'q':q_table,
            'batch_size':1,
            'buffer_size':10000,
            'gamma':0.9,
            'update_freq':1,
            'epsilon_lower':0.05,
            'epsilon_upper':0.8,
            'epsilon_decay_freq':200,
            'lr_lower':0.05,
            'lr_upper':0.5,
            'lr_decay_freq':200,
            'save_freq':50,
            'TDn': 10
        })
        tr = 0
        for iter in range(20):
            state = env.reset()
            done = False
            episode_reward = 0
            while not done:
                action = trainer.greedy(state)
                state, reward, done, _ = env.step(action)
                episode_reward += reward
                if (episode_reward > 500): break
                #env.render()
            if (episode_reward > 500): episode_reward = 0
            #preans = max(preans,episode_reward)
            #if (len(ans) > 0): episode_reward = max(episode_reward,max(ans))
            #ans.append(episode_reward)
            #preans_.append(preans)
            #env.close()
            tr+=episode_reward
            #if (len(ans) == 100): break
        tr/=20.0
        ans.append(tr)
        preans = max(preans,tr)
        preans_.append(preans)
        #if (len(ans) == 20): break
    ans_.append(preans_)
    ax.plot([i for i in range(len(ans))],ans)

ax.legend(indexes,loc='upper right')
plt.xlabel('Episodes')

plt.savefig("./pngs/result200.png")
plt.show()

fig1,ax1 = plt.subplots()
for i in range(len(ans_)):
    ax1.plot([j for j in range(len(ans_[i]))],ans_[i])
ax1.legend(indexes,loc='upper right')
plt.xlabel('Episodes')
plt.savefig("./pngs/result_premax_200.png")
plt.show()