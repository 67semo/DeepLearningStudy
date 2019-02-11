# Minimizing Cost 01
# use graph ploting

import tensorflow as tf
import matplotlib.pyplot as plt

# X and Y data
x_train = [1, 2, 3]
y_train = [1, 2, 3]

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.placeholder(tf.float32)
# Our hypothesis X*W
hypothesis = X * W

# cost/Loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Launch the graph in a session.
sess = tf.Session()
# Initializes gobal variables in the graph.
sess.run(tf.global_variables_initializer())

# Variables for plotting cost function
W_val = []
cost_val = []
for i in range(-30,50):
    feed_W = i*0.1
    curr_cost, curr_W = sess.run([cost, W], feed_dict={X:x_train, Y:y_train, W:feed_W})
    W_val.append(curr_W)
    cost_val.append(curr_cost)

# Show the cost function
plt.plot(W_val, cost_val)
plt.show()
