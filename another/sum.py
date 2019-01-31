import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 

class sumshueu():
    def __init__(self):
        self.state = 0.0
        self.reward_arr = np.array([
            [1, -1],
            [0, 0],
            [-1, 1]
        ])

        self.num_state = self.reward_arr.shape[0]
        self.num_action = self.reward_arr.shape[1] 

    def initState(self):
        # tmp = np.random.rand() 
        # if tmp < 0.3:
        #     self.state = 0.3
        # elif tmp < .6:
        #     self.state = 0.6
        # elif tmp < .9:
        #     self.state = 0.9
        self.state = np.random.randint(0, self.num_state)
        return self.state 
    
    def getState(self):
        return self.state

    def breath(self, _action):
        print(_action)
        reward = self.reward_arr[self.state, _action]
        if _action == 1:
            self.state -= 1
        elif _action == 0:
            self.state += 1
        if self.state < 0:
            self.state = 0
        elif self.state > 2:
            self.state = 2
        return reward
    
class agent():
    def __init__(self, _LR, _state_size):
        self.state_input = tf.placeholder(shape = [1], dtype = tf.int32)
        input_onehot = tf.one_hot(self.state_input, _state_size) 
        
        self.w0 = tf.Variable(tf.ones([_state_size, 1]), tf.float32)        
        self.n0 = tf.matmul(self.w0, input_onehot)
        self.activation = tf.nn.sigmoid(self.n0) 
        self.output = tf.reshape(self.activation, [-1])
        self.decided = tf.arg_max(self.output, 0) 

        self.reward_holder = tf.placeholder(shape = [1], dtype = tf.float32) 
        self.action_holder = tf.placeholder(shape = [1], dtype = tf.int32) 
        self.resp = tf.slice(self.output, self.action_holder, [1])
        self.loss = -(tf.log(self.resp) * self.reward_holder)
        # self.loss = -((self.resp - self.reward_holder) ** 2) / 2

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=_LR)
        self.update = optimizer.minimize(self.loss) 
    def predict(self, _session, _state_in):
        o, d = _session.run([self.output, self.decided], feed_dict = {self.state_input:_state_in})
        if d > .5:
            d = 1
        else:
            d = 0
        print(self.w0)
        return o, [d]
    
    def train(self, session, r, a, s_in):
        feed_dict = {
            self.reward_holder:r,
            self.action_holder:a,
            self.state_input:s_in
        }
        _, loss = session.run([self.update, self.loss], feed_dict=feed_dict)
        return loss

pye = sumshueu() 
myAgent = agent(1e-3, pye.num_state) 

total_eps = 20000

means = []
rewards = []

for t in range(20):
    init = tf.global_variables_initializer()
    pye.initState()
    
    with tf.Session() as sess:
        sess.run(init) 
        i = 0
        while i < total_eps:
            s = [pye.getState()]
            
            _, action = myAgent.predict(sess, s)

            reward = pye.breath(action)
            
            loss = myAgent.train(sess, reward, action, s) 
            for i in range(s[0]):
                print(")))")
