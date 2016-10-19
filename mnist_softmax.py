#download and read in the MNIST data set
#mnist.train has 55,000 data points
#mnist.test has 10,000 data points
#mnist.validation has 5000 data points
#mnist.train.images has the images - is a tensor of size [dataset, 28x28] OR [55000, 728]
#mnist.train.labels has the labels for the images - is an array of floats [55000, 10]
from tensorflow.examples.tutorials.mnist import input_data
#one_hot is 0 in most dimensions, 1 in a single dimension
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#softmax - good for assigning probabilities that something is one of many somethings
#weights - negative if evidence against image being in class; positive if evidence in favor
#bias - some things are more independent of input

import tensorflow as tf
#create a placeholder for computations - None means you can input any number of data points
x = tf.placeholder(tf.float32, [None,784])
#create Variables - modifiable tensor initialized to 0, 10 is number of classes
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
#define model
#y is predicted probability distribution
y = tf.nn.softmax(tf.matmul(x, W) + b)
#y_ is a placeholder
y_ = tf.placeholder(tf.float32, [None,10])
#cross_entropy measure how inefficient predictions are
#tf.log computes log of each element of y, multiply by y_
#tf.reduce_sum adds elements in second dimension of y bc of reduction_indices param
#tf.reduce_mean computes mean over all examples in batch
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
#training alogirthm - minimize cross_entropy using a gradient descent algorithm
#learning rate is 0.5
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#for value in [x, W, b, y, y_, cross_entropy]:
#  tf.scalar_summary(value.op.name, value)

#initialize variables created - does not run anything
init = tf.initialize_all_variables()
#launch session, run variable initialization
sess = tf.Session()

#playing with getting TensorBoard working
#summaries = tf.merge_all_summaries()
summary_writer = tf.train.SummaryWriter('log_mnist_softmax_stats', sess.graph)

sess.run(init)
#training loop - small batches of random data + gradient descent = stochastic gradient descent
for i in range(1000):
    #summary_writer.add_summary(sess.run(summaries), i)
    #for each step, get one batch of 100 data points from training set
    batch_xs, batch_ys = mnist.train.next_batch(100)
    #put batches in placeholders
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

#evaluating model
#tf.argmax gives index of highest entry in a tensor along an axis
#tf.argmax y,1 is label model assigns, y_,1 is the actual label; tf.equal checks, output is T/F
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
#cast T/F to floating point #s
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#print accuracy on test data
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

