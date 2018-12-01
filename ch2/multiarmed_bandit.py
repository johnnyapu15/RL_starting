import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt

# Set bandit. (Environment) 
bandit_values = [0.2, 0, -0.2, -2]
num_arms = bandit_values.__len__()
def pullBandit(bandit):
    result = np.random.randn(1) 
    if result > bandit:
        return 1
    else:
        return -1

# Set Agent.

tf.reset_default_graph() 

weights = tf.Variable(tf.ones([num_arms]))
output = tf.nn.softmax(weights) 

reward_holder = tf.placeholder(shape = [1], dtype = tf.float32) 
# action_holder is the index of action.
action_holder = tf.placeholder(shape = [1], dtype = tf.int32) 

res_output = tf.slice(output, action_holder, [1]) 
loss = -(tf.log(res_output) * reward_holder) 

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-4) 
update = optimizer.minimize(loss)

total_episodes = 10000

total_reward = np.zeros(num_arms)

init = tf.global_variables_initializer() 

outputs = []
with tf.Session() as sess: 
    sess.run(init)  
    i = 0
    while i < total_episodes:
        # Boltzman distribution 
        actions = sess.run(output) # .22, .5, .26, .02
        a = np.random.choice(actions, p = actions) # Let 0.26
        action = np.argmax(actions == a) # the index of 0.26 == 2
        ww1 = sess.run(weights) 
        if i % 2 == 0:
            outputs.append(actions)
        reward = pullBandit(bandit_values[action])

        ud, resp, ww, l = sess.run([update, res_output, weights, loss], \
            feed_dict = {reward_holder:[reward], action_holder:[action]}) 
        total_reward[action] += reward 

        if i % 500 == 0:
            print('When iteration: ' + str(i))
            print(" * Output of network: " + str(actions))
            print(" * Loss: " + str(l))
            print(" * Delta of weights: " + str(ww - ww1))
            print(" * Reward: " + str(total_reward))
        i += 1  

# Policy: probability of actions
plt.bar(np.arange(outputs[-1].__len__()), outputs[-1])
plt.show()