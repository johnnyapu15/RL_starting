# Annotated by johnnyapu15
# chapter 4. MDP python code by arthur juliani


# The cartpole.v0 problem description:
## https://github.com/openai/gym/wiki/CartPole-v0
import tensorflow as tf 
import tensorflow.contrib.slim as slim
import numpy as np 
import gym 
import matplotlib.pyplot as plt 



env = gym.make('CartPole-v0') 

gamma = 0.99 

def disc_rewards(r):
    disc_r = np.zeros_like(r) 
    running_add = 0 
    for t in range(0, r.size)[::-1]: 
        running_add = running_add * gamma + r[t] 
        disc_r[t] = running_add 
    return disc_r

class agent():
    def __init__(self, lr, s_size, a_size, h_size):
        # h_size: hidden
        self.state_in = tf.placeholder(shape=[None, s_size], dtype = tf.float32) 

        # Hidden layer
        # w0 = tf.Variable(tf.ones([s_size, h_size]), tf.float32)
        # b0 = tf.Variable(tf.zeros([h_size]), tf.float32)
        # o0 = tf.nn.relu(tf.nn.matmul(self.state_in, w0) + b0) 
        h = slim.fully_connected(self.state_in, h_size, biases_initializer=None, \
            activation_fn = tf.nn.relu) 
        
        # # Output layer
        # w1 = tf.Variable(tf.ones([h_size, a_size]), tf.float32)
        # b1 = tf.Variable(tf.zeros([a_size]), tf.float32)
        # self.o1 = tf.nn.softmax(tf.nn.matmul(o0, w1) + b1)
        self.o = slim.fully_connected(h, a_size, biases_initializer=None, \
            activation_fn = tf.nn.softmax)

        self.decided = tf.arg_max(self.o, 1) 

        self.reward_holder = tf.placeholder(shape=[None], dtype = tf.float32) 
        self.action_holder = tf.placeholder(shape=[None], dtype = tf.int32) 

        self.indexes = tf.range(0, tf.shape(self.o)[0]) * tf.shape(self.o)[1] \
            + self.action_holder 
        self.resp_outputs = tf.gather(tf.reshape(self.o, [-1]), self.indexes)

        self.loss = -tf.reduce_mean(tf.log(self.resp_outputs) * self.reward_holder) 

        tvars = tf.trainable_variables() 
        self.gradient_holders = [] 
        for idx, var in enumerate(tvars):
            placeholder = tf.placeholder(tf.float32, name = str(idx) + '_holder') 
            self.gradient_holders.append(placeholder) 
        
        self.gradients = tf.gradients(self.loss, tvars) 

        optimizer = tf.train.AdamOptimizer(learning_rate = lr) 
        self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders, tvars)) 

tf.reset_default_graph() 

# Output: 2 class. (by softmax)
# Learning_rate / state_size / action_size / hidden layer nodes
myAgent = agent(1e-2, 4, 2, 8) 

total_ep = 5000
max_ep = 999
update_frequency = 5 

init = tf.global_variables_initializer() 

with tf.Session() as sess: 
    sess.run(init) 
    i = 0 
    total_reward = []
    total_length = [] 

    # Get trainable variable array.
    gradBuff = sess.run(tf.trainable_variables()) 
    for idx, grad in enumerate(gradBuff): 
        # gradBuff is array!
        gradBuff[idx] = grad * 0 
    
    while i < total_ep: 
        # How descript the state 's'
        s = env.reset() 
        env.render()
        running_reward = 0
        ep_history = []
        for j in range(max_ep):
            # Choice the action 
            a_dist = sess.run(myAgent.o, feed_dict = {myAgent.state_in:[s]})
            a = np.random.choice(a_dist[0], p = a_dist[0]) 
            a = np.argmax(a_dist == a) 

            # What is this values?
            # https://gym.openai.com/docs/#observations
            
            # s1: next state
            # r: reward
            # d: is done?
            # _: info. (see docs above)

             
            s1, r, d, _ = env.step(a) 
            #print(s1)
            ep_history.append([s,a,r,s1]) 
            s = s1
            running_reward += r 
            if d == True:
                # print()
                # print()
                # Update the network 
                # d is 'end' value. 
                ep_history = np.array(ep_history) 
                
                # [:,2] = rewards. (column '2')
                # Let's discount rewards.
                ep_history[:, 2] = disc_rewards(ep_history[:, 2])
                
                feed_dict = {
                    myAgent.reward_holder:ep_history[:, 2],
                    myAgent.action_holder:ep_history[:, 1],
                    # Why do 'transpose' the states?
                    myAgent.state_in:np.vstack(ep_history[:, 0])
                    }
                
                grads = sess.run(myAgent.gradients, feed_dict = feed_dict) 

                # gradBuff is the rollout(experience trace).
                for idx, grad in enumerate(grads):
                    gradBuff[idx] += grad 
                
                # is it ok to update?
                if i % update_frequency == 0 and i != 0: 
                    feed_dict = dictionary = dict(zip(myAgent.gradient_holders, \
                        gradBuff)) 
                    _ = sess.run(myAgent.update_batch, feed_dict = feed_dict) 

                    # grad is array.
                    for idx, grad in enumerate(gradBuff):
                        gradBuff[idx] = grad * 0
                total_reward.append(running_reward) 
                total_length.append(j) 
                # Let's go to next episode.
                break
        if i % 100 == 0:
            print(np.mean(total_reward[-100:]))
        i += 1





        