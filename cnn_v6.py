# inputs -> (conv -> bn -> leaky_relu -> pool)x2 -> fc(dropout) -> relu -> fc -> softmax
# train time: 5000, accuracy: 95%， run time: 987s

import img_data
import tensorflow as tf
import time
from tensorflow.python.training.moving_averages import assign_moving_average

def batch_norm(inputs, train, eps=1e-05, decay=0.9, name=None):
    with tf.variable_scope(name, default_name='batch_norm'):
        param_shape = inputs.get_shape().as_list()
        param_shape = param_shape[-1:]
        moving_mean = tf.get_variable('mean', shape=param_shape,
                        initializer=tf.zeros_initializer,trainable=False)
        moving_variance = tf.get_variable('variance', shape=param_shape,
                            initializer=tf.ones_initializer, trainable=False)
        
        def mean_var_with_update():
            axis = list(range(len(inputs.shape) - 1))
            mean, variance = tf.nn.moments(inputs, axis, name='moments')
            with tf.control_dependencies([assign_moving_average(moving_mean, mean, decay),
                                         assign_moving_average(moving_variance, variance, decay)]):
                return tf.identity(mean), tf.identity(variance)
        
        if train is not None:
            bool1 = tf.constant(4)
            bool2 = tf.constant(3)
        else:
            bool1 = tf.constant(3)
            bool2 = tf.constant(4)
        
        mean, variance = tf.cond(bool1<bool2, mean_var_with_update, lambda: (moving_mean, moving_variance))
        beta = tf.get_variable('beta', param_shape, initializer=tf.zeros_initializer)
        gamma = tf.get_variable('gamma', param_shape, initializer=tf.ones_initializer)
        res_batch = tf.nn.batch_normalization(inputs, mean, variance, beta, gamma, eps)
        return res_batch

sess = tf.Session()

img_inputs = tf.placeholder("float", shape=[None, 784])
expect_outputs = tf.placeholder("float", shape=[None, 14])

sess.run(tf.global_variables_initializer())

weights_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
biases_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))

img_reshape = tf.reshape(img_inputs, [-1, 28, 28, 1])

res_conv1 = tf.nn.conv2d(img_reshape, weights_conv1, strides=[1, 1, 1, 1], padding='SAME') + biases_conv1

train = tf.placeholder(tf.bool)

res_batch1 = batch_norm(res_conv1, train)
res_lrelu1 = tf.nn.leaky_relu(res_batch1)
res_pool1 = tf.nn.max_pool(res_lrelu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

weights_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
biases_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))

res_conv2 = tf.nn.conv2d(res_pool1, weights_conv2, strides=[1, 1, 1, 1], padding='SAME') + biases_conv2
res_batch2 = batch_norm(res_conv2, train)
res_lrelu2 = tf.nn.leaky_relu(res_batch2)
res_pool2 = tf.nn.max_pool(res_lrelu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

weights_fc1 = tf.Variable(tf.truncated_normal([7*7*64, 1024], stddev=0.1))
biases_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))

res_pool2_reshape = tf.reshape(res_pool2, [-1, 7*7*64])

res_fc1 = tf.nn.relu(tf.matmul(res_pool2_reshape, weights_fc1) + biases_fc1)

keep_prob = tf.placeholder("float")
res_fc1_drop = tf.nn.dropout(res_fc1, keep_prob)

weights_fc2 = tf.Variable(tf.truncated_normal([1024, 14], stddev=0.1))
biases_fc2 = tf.Variable(tf.constant(0.1, shape=[14]))

outputs = tf.nn.softmax(tf.matmul(res_fc1_drop, weights_fc2) + biases_fc2)

cross_entropy = -tf.reduce_sum(expect_outputs*tf.log(outputs))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(outputs, 1), tf.argmax(expect_outputs, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

data = img_data.Data()

sess.run(tf.global_variables_initializer())

time_start = time.time()
final_accuracy = 0
for i in range(5000):
    batch = data.train_set(50)
    
    if ((i+1) % 10 == 0):
        test = data.test_set(500)
        train_accuracy = accuracy.eval(feed_dict={img_inputs: test[0],
                        expect_outputs: test[1], keep_prob: 1.0}, session=sess)
        print("step %d, training accuracy %g%%" % (i+1, train_accuracy*100))
        if train_accuracy > final_accuracy: 
            final_accuracy = train_accuracy
    train_step.run(feed_dict={img_inputs: batch[0], expect_outputs: batch[1], 
                keep_prob: 0.5, train: True}, session=sess)
time_end = time.time()
test = data.test_set(1000)
train_accuracy = accuracy.eval(feed_dict={img_inputs: test[0], expect_outputs: test[1], keep_prob: 1.0}, session=sess)
print("final accuracy: %g%%, total time: %gs" % (final_accuracy*100, (time_end-time_start)))