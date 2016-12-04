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
tf.reset_default_graph()
logs_path = "log_mnist_adv2"
#allow interleaving of ops ELSE build computation graph first, then start session and launch graph
#computation graph - defines operations outside of Python instead of switching back and forth a bunch
sess = tf.InteractiveSession()

#### functions 
#for weight initialization - small amount of noise added to prevent 0 gradients & symmetry breaking
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
#relu - initialize with slightly positive bias to prevent dead neurons
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
#convolutions - stride of 1, zero padded (output = input)
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
#pooling - max pooling over 2x2 blocks
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

###super simple, no learning version.  Just MNIST on Linear adaptive filter
def LAF():
    #create a placeholder for computations - None means you can input any number of data points
    x = tf.placeholder(tf.float32, [None,784])
    #y_ is a placeholder for target output classes
    y_ = tf.placeholder(tf.float32, [None,10])
    #create Variables - modifiable tensor initialized to 0, 10 is number of classes
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    #takes initial values, assigns them to variables
    sess.run(tf.initialize_all_variables())

    #define model
    #y is predicted probability distribution
    #x * W + b, then compute softmax probabilities
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    #cross_entropy measure how inefficient predictions are (loss function)
    #tf.log computes log of each element of y, multiply by y_
    #tf.reduce_sum adds elements in second dimension of y bc of reduction_indices param
    #tf.reduce_mean computes mean over all examples in batch
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    #training alogirthm - minimize cross_entropy using a gradient descent algorithm
    #learning rate is 0.5
    #Note: this just defines (ie. computational graph).  It runs nothing itself
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    #training loop - small batches of random data + gradient descent = stochastic gradient descent
    for i in range(1000):
        #for each step, get one batch of 100 data points from training set
        batch = mnist.train.next_batch(100)
        #put batches in placeholders using feed_dict - x is images, y_ is labels
        sess.run(train_step, feed_dict={x: batch[0], y_: batch[1]})

    #evaluating model
    #tf.argmax gives index of highest entry in a tensor along an axis
    #tf.argmax y,1 is label model assigns, y_,1 is the actual label; tf.equal checks, output is T/F
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    #cast T/F to floating point #s
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #print accuracy on test data
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

def conv_layers():
    ##########
    ##References for Tensorboard
    ##https://ischlag.github.io/2016/06/04/how-to-use-tensorboard/
    ##http://robromijnders.github.io/tensorflow_basic/
    ##########
    with tf.name_scope('input'):
        #create a placeholder for computations - None means you can input any number of data points
        x = tf.placeholder(tf.float32, [None,784], name="image_input")
        #y_ is a placeholder for target output classes
        y_ = tf.placeholder(tf.float32, [None,10], name="label_input")
    with tf.name_scope('weights'):
        #create Variables - modifiable tensor initialized to 0, 10 is number of classes
        W = tf.Variable(tf.zeros([784, 10]))
    with tf.name_scope('biases'):
        b = tf.Variable(tf.zeros([10]))
        
    #print original image
    #tf.image_summary("Original_Image", x)
    #layer 1 - will compute 32 features for each 5x5 patch - tensor[patch_height, patch_width, input channels, output channels]
    #bias vector has component for each output channel
    with tf.name_scope('Conv1'):
        #reshape x to a 4D tensor - -1, image_height, image_width, #color_channels
        x_image = tf.reshape(x, [-1, 28, 28, 1])
        tf.image_summary("Reshaped_Image", x_image)
        W_conv1 = weight_variable([5, 5, 1, 32])
        #tf.image_summary("vis_conv1", W_conv1)
        b_conv1 = bias_variable([32])
        #convolve x image with weight tensor, add bias, apply ReLU, perform max pool
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    with tf.name_scope('Pool1'):
        #pooling is reducing image size to 14x14
        h_pool1 = max_pool_2x2(h_conv1)

    #layer 2 - 64 features per 5x5 patch
    with tf.name_scope('Conv2'):
        W_conv2 = weight_variable([5, 5, 32, 64])
        #tf.image_summary("vis_conv2", W_conv2)
        b_conv2 = bias_variable([64])
        #pooling will reduce image size to 7x7
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    with tf.name_scope('Pool2'):
        h_pool2 = max_pool_2x2(h_conv2)
        #tf.image_summary("An_Image", x_image)

    #densely connected / fully-connected layer - 1024 neurons used here
    with tf.name_scope('FullConnect'):
        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])
        #reshape tensor from h_pool2 into batch of vectors, multiply by weight matrix, add bias, apply ReLU
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    #apply dropout HERE, before readout layer - turn on during training, off during testing
    with tf.name_scope('Dropout'):
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    
    with tf.name_scope('softmax'):
        #readout layer / softmax layer
        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])
        y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    #cross_entropy measure how inefficient predictions are (loss function)
    #tf.log computes log of each element of y, multiply by y_
    #tf.reduce_sum adds elements in second dimension of y bc of reduction_indices param
    #tf.reduce_mean computes mean over all examples in batch
    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
    with tf.name_scope('Train'):
        #train with AdamOptimizer
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    with tf.name_scope('Accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    #summary of cost / loss and accuracy
    tf.scalar_summary("cost", cross_entropy)
    tf.scalar_summary("accuracy", accuracy)
    tf.histogram_summary("W_fc2", W_fc2)
    tf.histogram_summary("b_fc2", b_fc2)
        
    #merge summaries into single op
    summary_op = tf.merge_all_summaries()

    #have fully defined computational model, now initialize all Variables and start training
    sess.run(tf.initialize_all_variables())
    
    #create log writer
    summary_writer = tf.train.SummaryWriter(logs_path, graph=tf.get_default_graph())
    
    #run it for 20,000 iterations
    for i in range(2000):
        batch = mnist.train.next_batch(50)
        if i%100 == 0:
            #keep_prob parameter controls dropout rate
            train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g"%(i, train_accuracy))
        #train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        _, summary = sess.run([train_step, summary_op], feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        summary_writer.add_summary(summary, i * 50)    
    print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


#LAF()
conv_layers()

