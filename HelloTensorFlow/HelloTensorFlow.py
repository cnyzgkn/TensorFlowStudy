import tensorflow as tf
import numpy as np

#计算 a=(b+c)*(c+2)
# 首先，创建一个TensorFlow常量=>2
const = tf.constant(2.0, name='const')

# 创建TensorFlow变量b和c
b = tf.Variable(2.0, name='b')
c = tf.Variable(1.0, dtype=tf.float32, name='c')

# 创建operation
d = tf.add(b, c, name='d')
e = tf.add(c, const, name='e')
a = tf.multiply(d, e, name='a')

# 创建TensorFlow变量f，使f可以接收任何值
#计算 g=a*f
f = tf.placeholder(tf.float32, [None, 1], name='f')
g = tf.multiply(a, f, name='g')

# 定义init operation
init_op = tf.global_variables_initializer()

# session
with tf.Session() as sess:
    # 2. 运行init operation
    sess.run(init_op)
    # 计算
    a_out = sess.run(a)
    print("Variable a is {}".format(a_out))
    g_out = sess.run(g, feed_dict={f: np.arange(0, 10)[:, np.newaxis]})
    print("Variable f is {}".format(g_out))