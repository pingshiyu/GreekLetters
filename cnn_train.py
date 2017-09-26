'''
Created on 13 Sep 2017

@author: pingshiyu
'''

'''
    Train and obtain a CNN model on the greek_letters dataset.
    The dataset used this time is around ~240k in size.
    The CNN here will have structure:
    
    input -> conv(6*6, 40 chnls + 2x2 pool) -> conv(5*5, 40 chnls + 2x2 pool)
    -> fc1 -> output
    
    All layers have ReLU activation.
'''

# for data feeding
from feeder import Feeder

# for NNs
import tensorflow as tf

# pre-made custom layers
from tensorflow_layers import fc_layer, conv_layer, flatten_2d

# constants:
CLASSES = 24
IMG_SIZE = 40
IMG_SIZE_FLAT = IMG_SIZE*IMG_SIZE
TRAIN_BATCH_SIZE = 128
MODELNUM = 3
TENSORBOARD_DIR = './tmp/{}/'.format(MODELNUM)

# load in the data saved in './data/warped_40x40/warped_data_240k.csv'
data = Feeder(file_path = './data/warped_40x40/warped_data_240k.csv',
              classes = CLASSES)

def feed_dict(train = True, all_test_data = False):
    '''
    Return the feed_dict for train or testing mode (since it is called a lot)
    ``all_test_data`` returns all of the testing data, which may be slow to
    evaluate.
    '''
    if train:
        xs, ys = data.next_batch(TRAIN_BATCH_SIZE)
        p = 0.5
    # here we use testing data, but do we use ``all_test_data``?
    elif not all_test_data:
        xs, ys = (data.test[0])[:500], (data.test[1])[:500]
        p = 1.0
    else: # use all test data
        xs, ys = data.test
        p = 1.0
        
    return {X: xs, y: ys, keep_prob: p}

if __name__ == '__main__':
    sess = tf.InteractiveSession()
    
    with tf.name_scope('inputs'):
        # define the X feature inputs
        X = tf.placeholder(tf.float32, [None, IMG_SIZE_FLAT])
        x_conv_input = tf.reshape(X, [-1, IMG_SIZE, IMG_SIZE, 1])
        
        # define y, the classes of our feature
        # note: y is 1d to take advantage of tensorflow's C++ one-hot conversion
        y = tf.placeholder(tf.uint8, [None])
        y_one_hot = tf.one_hot(indices = y, depth = CLASSES)
        
        # actual y classes:
        y_actual_class = tf.transpose(tf.cast(y, tf.int64))
        
        # take a peek at what the input is like
        tf.summary.image('input_img', x_conv_input)
        
    conv_layer1 = conv_layer(x_conv_input, 
                             input_channels = 1, 
                             output_channels = 40, 
                             filter_dimension = 6,
                             with_pooling = True,
                             padding = 'VALID',
                             name = 'conv1')
    
    conv_layer2 = conv_layer(conv_layer1,
                             input_channels = 40,
                             output_channels = 80,
                             filter_dimension = 5,
                             with_pooling = True,
                             padding = 'VALID',
                             name = 'conv2')
    
    # fc1_features should be IMG_DIM*IMG_DIM*16/(4^2) = IMG_DIM*IMG_DIM
    fc1_input, fc1_features = flatten_2d(conv_layer2)
    # for dropout:
    keep_prob = tf.placeholder(tf.float32)
    fc1 = fc_layer(fc1_input, num_inputs = fc1_features, num_nodes = 1024,
                              use_relu = True,
                              with_dropout = True, keep_prob = keep_prob,
                              name = 'fc1')
    
    # this is the output layer. no activation as we will apply softmax
    fc2 = fc_layer(fc1, num_inputs = 1024, num_nodes = CLASSES,
                        name = 'output')
    
    # calculates accuracy of our prediction
    with tf.name_scope('accuracy'):
        predictions = tf.argmax(input = fc2, axis = 1)
        correct_predictions = tf.equal(predictions, y_actual_class)
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
        tf.summary.scalar('accuracy', accuracy)
    
    with tf.name_scope('loss_func'):
        # loss is defined as the average cross-entropy cost
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels = y_one_hot, logits = fc2))
        tf.summary.scalar('loss', loss)
    
    with tf.name_scope('train'):
        # keep track of global training step for learning rate decay.
        global_step = tf.Variable(0, trainable = False)
        init_learning_rate = 1e-3
        # we initialise learning with 1e-3, decaying once every 200 steps at a
        # rate of 0.96. By the end of 50k steps the learning rate will have 
        # decayed by a factor of ~3e-5 of the original
        learning_rate = tf.train.exponential_decay(init_learning_rate, 
                                                   global_step,
                                                   133, 
                                                   0.96, 
                                                   staircase = True,
                                                   name = 'learning_rate')
        # keep track of learning rate summary
        tf.summary.scalar('curr_learning_rate', learning_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step = global_step)
    
    # so we can save trained models
    saver = tf.train.Saver()
    
    # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(TENSORBOARD_DIR + 'train',
                                          sess.graph)
    test_writer = tf.summary.FileWriter(TENSORBOARD_DIR + 'test')
    tf.global_variables_initializer().run()
    
    # add graph to tensorboard
    train_writer.add_graph(sess.graph)
    
    # network construction is now complete - we now run it
    # necessary to initialise our variables
    sess.run(tf.global_variables_initializer())
    
    # train our network for a lot of generations
    for i in range(20000):
        summary, _ = sess.run([merged, optimizer], 
                              feed_dict=feed_dict(True))
        train_writer.add_summary(summary, i)
        
        if i%10 == 0:
            summary, acc = sess.run([merged, accuracy], 
                                    feed_dict = feed_dict(False))
            test_writer.add_summary(summary, i)
            print('Accuracy at step %s: %s' % (i, acc))
        
        # save the model (graph) once in a while
        if i%5000 == 0 and i>1:
            save_path = saver.save(sess, "./models/{1}/model_{0}.ckpt".format(i, MODELNUM))
            print("Model checkpoint saved in file: %s" % save_path)
            
    # after training: save our network and print out test accuracy
    test_acc = sess.run(accuracy, 
                        feed_dict = feed_dict(False, all_test_data=True))
    print('Training finished! Test accuracy: {}'.format(test_acc))
    
    # and save our final network
    save_path = saver.save(sess, "./models/{}/model_final.ckpt".format(MODELNUM))
    print("Model saved in file: %s" % save_path)
    
    train_writer.close()
    test_writer.close()
    sess.close()