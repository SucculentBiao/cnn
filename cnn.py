import tensorflow as tf 
import img_data

img_inputs = tf.placeholder('float', shape=[None, 784])
expec_outputs = tf.placeholder('float', shape=[None, 14])

weights_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
biases_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))

img_reshape = tf.reshape(img_inputs, [-1, 28, 28, 1])
res_conv1 = tf.nn.relu(tf.nn.conv2d(img_reshape, weights_conv1, 
                strides=[1, 1, 1, 1], padding='SAME') + biases_conv1)
res_pool1 = tf.nn.max_pool(res_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

weights_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
biases_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))

res_conv2 = tf.nn.relu(tf.nn.conv2d(res_pool1, weights_conv2, 
                strides=[1, 1, 1, 1], padding='SAME') + biases_conv2)
res_pool2 = tf.nn.max_pool(res_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

weights_fc1 = tf.Variable(tf.truncated_normal([7*7*64, 1024]))
biases_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))

res_pool2_reshape = tf.reshape(res_pool2, [-1, 7*7*64])
res_fc1 = tf.nn.relu(tf.matmul(res_pool2_reshape, weights_fc1) + biases_fc1)

keep_prob = tf.placeholder('float')
res_fc1_drop = tf.nn.dropout(res_fc1, keep_prob)

weights_fc2 = tf.Variable(tf.truncated_normal([1024, 14], stddev=0.1))
biases_fc2 = tf.Variable(tf.constant(0.1, shape=[14]))

outputs = tf.nn.softmax(tf.matmul(res_fc1_drop, weights_fc2) + biases_fc2)

cross_entropy = -tf.reduce_sum(expec_outputs*tf.log(outputs))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_pre = tf.equal(tf.argmax(outputs, 1), tf.argmax(expec_outputs, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pre, 'float'))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

data = img_data.Data()

for i in range(1000):
    batch = data.train_set(50)
    test = data.test_set(200)
    train_step.run(feed_dict={img_inputs: batch[0], 
                    expec_outputs: batch[1], keep_prob: 0.5}, session=sess)
    train_acc = accuracy.eval(feed_dict={img_inputs: test[0], 
                    expec_outputs: test[1], keep_prob: 1.0}, session=sess)
    print("train time: %d, accuracy: %f" % (i, train_acc))