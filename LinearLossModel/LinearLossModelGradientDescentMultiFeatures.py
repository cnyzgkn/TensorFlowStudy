import tensorflow as tf
import numpy as np

n_features = 4
m_examples = 47

x_in = tf.placeholder(tf.float32, [None, n_features], "x_in")
w = tf.Variable(tf.random_normal([n_features, 1]), name="w")
b = tf.Variable(tf.constant(0.1, shape=[]), name="b")
linear_model = tf.add(tf.matmul(x_in, w), b)

y_in = tf.placeholder(tf.float32, [None, 1], "y_in")
loss_op = tf.reduce_mean(tf.square(tf.subtract(y_in, linear_model)), name="loss")
train_op = tf.train.GradientDescentOptimizer(0.3).minimize(loss_op)

# 生成希望学习得到的数据
x_train = np.random.rand(m_examples, n_features)
w_true = np.random.rand(n_features, 1) * 100.0
b_true = np.random.rand(1) * 100.0 - 50.0
noise = np.random.rand(m_examples, 1) / 100.0
y_train = (np.matmul(x_train, w_true) + b_true) + noise

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for batch in range(1000):
        sess.run(train_op, feed_dict={
            x_in: x_train,
            y_in: y_train
        })
    w_computed = sess.run(w)
    b_computed = sess.run(b)

print("w computed [%s]" % ', '.join(['%.5f' % x for x in w_computed.flatten()]))
print("w actual   [%s]" % ', '.join(['%.5f' % x for x in w_true.flatten()]))
print("b computed %.3f" % b_computed)
print("b actual  %.3f" % b_true[0])
