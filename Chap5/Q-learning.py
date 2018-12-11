# Annotated by johnnyapu15
# chapter 5. Q learning python code by arthur juliani
# 20181211

import tensorflow as tf 
import gym 
import numpy as np 
import random 
import matplotlib.pyplot as plt 

LEARNING_RATE = 0.1

env = gym.make('FrozenLake-v0') 

tf.reset_default_graph() 

inputs1 = tf.placeholder(shape = [1, 16], dtype = tf.float32) 
W = tf.Variable(tf.random_uniform([16, 4], 0, 0.01))
Qout = tf.matmul(inputs1, W)
# GREEDY EXPLORER 
predict = tf.argmax(Qout, 1) 

tQ = tf.placeholder(shape = [1, 4], dtype = tf.float32) 
loss = tf.reduce_sum(tf.square(tQ - Qout)) 
trainer = tf.train.GradientDescentOptimizer(learning_rate = LEARNING_RATE)
updateModel = trainer.minimize(loss) 

init = tf.global_variables_initializer() 

GAMMA = 0.99 

e = 0.1 #EPSILLON
num_eps = 2000 

jList = []
rList = [] 

with tf.Session() as sess:
    sess.run(init) 
    for i in range(num_eps):
        s = env.reset() 
        rAll = 0
        d = False 
        j = 0

        while j < 99:
            j += 1
            a, allQ = sess.run([predict, Qout], \
                feed_dict = {inputs1:np.identity(16)[s:s+1]})
            # See np.identity(n) 
            
            # Epsillon greedy
            if np.random.rand(1) < e:
                a[0] = env.action_space.sample()
            snext, r, d, _ = env.step(a[0]) 

            # Set Qnext.
            Qnext = sess.run(Qout, feed_dict = {inputs1: np.identity(16)[snext:snext+1]})
            maxQnext = np.max(Qnext) 
            targetQ = allQ
            targetQ[0, a[0]] = r + GAMMA * maxQnext
            #targetQ[0, a[0]] += r + GAMMA*(maxQnext - targetQ[0, a[0]])
            _, Wnext = sess.run([updateModel, W], \
                feed_dict= {inputs1:np.identity(16)[s:s+1], tQ:targetQ})
            rAll += r 
            s = snext 
            if d == True:
                e = 1./((i / 50) + 10) 
                break 
        jList.append(j)
        rList.append(rAll) 

print("Percent of successful eps: " + str(sum(rList) / num_eps)) 

plt.plot(rList) 
plt.show() 
plt.plot(jList)
plt.show()
            