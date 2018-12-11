# Annotated by johnnyapu15
# chapter 5. Q table python code by arthur juliani
# 20181210

import gym
import numpy as np 
import matplotlib.pyplot as plt 

env = gym.make('FrozenLake-v0')

Q = np.zeros([env.observation_space.n, env.action_space.n]) 

LEARNING_RATE = 0.85 
GAMMA = 0.99 
num_eps = 2000

rList = []
for i in range(num_eps):
    s = env.reset() 
    rAll = 0
    d = False 
    j = 0
    while j < 99:
        j += 1
        # Select action greedy with noise
        # Is it 'boltzman distribution'? 
        ## No, it's the method 'epsillon greedy' like.
        a = np.argmax(Q[s, :] + np.random.randn(1, env.action_space.n) * (1. / (i + 1)))
        s1, r, d, _ = env.step(a) 
        # Update the Q-table
        # Q = (1 - lr)*Q + (lr)*(r + y*max(Q'))
        # Memorize the past about (1 - lr)
        Q[s, a] = Q[s, a] + LEARNING_RATE * (r + GAMMA * np.max(Q[s1, :]) - Q[s, a]) 
        rAll += r
        s = s1 
        if d == True:
            break 
    rList.append(rAll) 

print("Score over time: " + str(sum(rList)/num_eps)) 

print("Final Q-table values ")
print(Q)