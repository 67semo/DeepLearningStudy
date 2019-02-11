# Minimizing Cost 01
# tensorflower의 함수와 미분에의한 값비교

import tensorflow as tf

# X and Y data
x_train = [1, 2, 3]
y_train = [1, 2, 3]

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(5.)
# Our hypothesis X*W
hypothesis = X * W

# cost/Loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize: Gradient Descent using derivative:
learning_rate = 0.01
gradient = tf.reduce_mean((W * X - Y) * X * 2)
descent = W - learning_rate * gradient
update = W.assign(descent)

# Optimized by GradientDescentOptimiaer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
gvs = optimizer.compute_gradients(cost,[W])
apply_gradients = optimizer.apply_gradients(gvs)

# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

for step in range(100):
    sess.run(update, feed_dict={X:x_train, Y:y_train})
    print(step, sess.run(cost, feed_dict={X:x_train, Y:y_train}), sess.run(W))
