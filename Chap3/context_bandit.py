import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 


class contextual_bandit():
    def __init__(self):
        self.state = 0
        self.bandits = np.array([
            [0.2, 0, -0.0, -5],
            [0.1, -5, 1, 0.25],
            [-5, 5, 5, 5]
        ])
        self.num_bandits = self.bandits.shape[0] 
        self.num_actions = self.bandits.shape[1] 

    def getState(self):
        self.state = np.random.randint(0, self.num_bandits)
        return self.state 
    
    def pullArm(self, action):
        bandit_value = self.bandits[self.state, action] 
        result = np.random.randn(1) 
        if result > bandit_value:
            return 1
        else:
            return -1
    
    
class agent():
    def __init__(self, LEARNING_RATE, state_size, action_size):
        self.state_input = tf.placeholder(shape = [1], dtype = tf.int32) 
        input_onehot = tf.one_hot(self.state_input, state_size) 
#         self.w0 = tf.Variable(tf.truncated_normal([state_size, action_size]), tf.float32)
        #self.b0 = tf.Variable(tf.truncated_normal([action_size]), tf.float32)
        self.w0 = tf.Variable(tf.ones([state_size, action_size]), tf.float32)
        self.b0 = tf.Variable(tf.zeros([action_size]), tf.float32)
        
        # If there is bias, it can't train properly at second bandit case.  #####
        self.o0 = tf.nn.sigmoid(tf.matmul(input_onehot, self.w0))
        #self.o0 = tf.nn.sigmoid((input_onehot * self.w0) + self.b0)
        #self.output = self.o0
        
        self.softmax = tf.nn.softmax(self.o0)
        self.output = tf.reshape(self.softmax, [-1])
        self.decided = tf.arg_max(self.output, 0) 

        self.reward_holder = tf.placeholder(shape = [1], dtype = tf.float32) 
        self.action_holder = tf.placeholder(shape = [1], dtype = tf.int32) 
        self.resp = tf.slice(self.output, self.action_holder, [1])
        self.loss = -(tf.log(self.resp) * self.reward_holder) 
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE)
        self.update = optimizer.minimize(self.loss) 

    def predict(self, session, s_in):
        o, d = session.run([self.output, self.decided], feed_dict = {self.state_input:s_in})
        return o, d
    
    def train(self, session, r, a, s_in):
        feed_dict = {
            self.reward_holder:r,
            self.action_holder:a,
            self.state_input:s_in
        }
        _, loss = session.run([self.update, self.loss], feed_dict=feed_dict)
        return loss

cBandit = contextual_bandit() 
myAgent = agent(1e-3,cBandit.num_bandits, cBandit.num_actions) 

total_episodes = 20000


e = 0.1 # Like Random walker.


means = []
rewards = []
for t in range(20):
    tmp_rewards = []
    tmp_total = np.zeros([cBandit.num_bandits])
    total_reward = np.zeros([cBandit.num_bandits, cBandit.num_actions]) 
    init = tf.global_variables_initializer() 

    with tf.Session() as sess:
        sess.run(init) 
        i = 0
        while i < total_episodes:
            s = [cBandit.getState()]
            if np.random.rand(1) < e: 
                action = [np.random.randint(cBandit.num_actions)] 
            else:
                pred, action = myAgent.predict(sess, s)
                action = [action]

            reward = [cBandit.pullArm(action)]

            loss = myAgent.train(sess, reward, action, s) 
            if i % 100 == 0:
                
                tmp_rewards.append(np.mean(np.mean(total_reward, axis=1) - tmp_total))
                tmp_total = np.mean(total_reward, axis=1)
            total_reward[s, action] += reward
            if i == total_episodes-1:
                print("Testcase " + str(t) + "---------------------------")
                print("Iteration " + str(i))
                print("Loss: " + str(loss)) 
#                 for a in range(cBandit.num_bandits):
#                     pred, _ = myAgent.predict(sess, [a])
#                     print(pred)
                print("total rewards: " + str(np.mean(total_reward, axis=1)))
            i += 1
        means.append(np.mean(total_reward, axis=1))
        rewards.append(tmp_rewards)
print()
print()
print("---------------------------------------------------")      
print("result: " + str(np.mean(means, axis = 0)))
print("mean: " + str(np.mean(means)))
with_mean = means
with_soft = np.mean(rewards, axis = 0)
            
        