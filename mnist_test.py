'''
Created on 12 Aug 2017

@author: pings
'''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

h_l1_nodes = 300
h_l2_nodes = 100

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

X = tf.placeholder('float', [None, 28*28])
y = tf.placeholder('float', [None, 10])

def nn_feedforward(batch):
    '''
        Returns a tensor of the activations of last layer of
        network for each of the data points, in the same 
        as order given.
    '''
    # layers will be matrices of weights such that activation
    # to the previous layer, multiplied by the layer gives the 
    # activation of the current layer
    input_hl1 = {'weights': tf.Variable(tf.random_normal([28*28, h_l1_nodes])),
                 'biases': tf.Variable(tf.random_normal([h_l1_nodes]))}
    hl1_hl2 = {'weights': tf.Variable(tf.random_normal([h_l1_nodes, h_l2_nodes])),
               'biases': tf.Variable(tf.random_normal([h_l2_nodes]))}
    hl2_output = {'weights': tf.Variable(tf.random_normal([h_l2_nodes, 10])),
                   'biases': tf.Variable(tf.random_normal([10]))}
    
    # data_point is activation of input_layer
    hl1_a = tf.nn.softplus(
        tf.matmul(batch, input_hl1['weights']) + input_hl1['biases'])
    hl2_a = tf.nn.softplus(
        tf.matmul(hl1_a, hl1_hl2['weights']) + hl1_hl2['biases'])
    output_logits = tf.matmul(hl2_a, hl2_output['weights']) + hl2_output['biases']
    
    # return logits of activation of final layer
    return output_logits

def nn_train(epochs = 10, batch_size = 100, rate = 0.001,
             print_accuracy = True):
    '''
        Train neural network with the data given for specified
        epoch number, also specifying batch_size
        data is 
    '''
    prediction = nn_feedforward(X)
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            labels = y,
            logits = prediction))
    train_step = tf.train.AdamOptimizer(rate).minimize(loss)
    
    # to evaluate accuracy of our model
    test_xs = mnist.test.images
    test_ys = mnist.test.labels
    correct = tf.equal(tf.argmax(prediction, 1),
                       tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch_no in range(1, epochs+1):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples / batch_size)):
                xs, ys = mnist.train.next_batch(batch_size)
                batch_loss, _ = sess.run([loss, train_step],
                                         feed_dict = {X: xs, y: ys})
                epoch_loss += batch_loss
            print('epoch %i loss: %f' % (epoch_no, epoch_loss))
            if print_accuracy: # may slow down code
                print('accuracy at end of epoch:',
                      sess.run(accuracy, 
                               feed_dict = {X: test_xs, y: test_ys}))
            
        # evaluate accuracy of final model
        print(sess.run(accuracy, feed_dict = {X: test_xs, y: test_ys}))

nn_train(epochs = 10, batch_size = 100, rate = 0.001)