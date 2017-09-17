'''
Created on 15 Sep 2017

@author: pingshiyu
'''

import tensorflow as tf

# for tensorboard
from tensorboard_tools import variable_summaries, put_kernels_on_grid

'''
    Here are some pre-built layers I made to speed up development.
    fc_layer: Takes in an input tensor (2D) + params;
              Returns a tensor of its activations
    conv_layer: Takes in an input tensor (4D) + params;
                Returns a 4D tensor of its activations
'''

def weight_variable(shape):
    '''
    Create a tf weight variable, initialised from normal distribution, 
    according to ``shape`` 
    '''
    return tf.Variable(tf.truncated_normal(shape, stddev = 0.05))

def bias_variable(size):
    '''
    Biases are 1d, with ``size`` for its length
    '''
    return tf.Variable(tf.constant(0.12, shape = [size], dtype = tf.float32))

def flatten_2d(tensor):
    '''
    Takes in a 4d tensor (assumed) and flattens it to become a 2d tensor fit
    for fully-connected layers.
    
    Returns also the number of features calculated, along with, of course,
    the flattened layer
    '''
    shape = tensor.get_shape()
    num_features = shape[1:].num_elements()
    
    return tf.reshape(tensor, [-1, num_features]), num_features

def conv_layer(X, input_channels = 1, output_channels = 32,
                  filter_dimension = 5,
                  with_pooling = True,
                  name = None):
    '''
    Puts the data input through the convolutional layer, specified by its 
    parameters.
    ``with_pooling`` set to True => turns on 2x2 pooling on exiting the 
    layer.
    
    input_channels: int
    output_channels: int
    filter_dimension: int specifying the side length of the square filter
    with_pooling: boolean
    name: String
    
    Returns the ``X`` after being transformed by the layer
    '''
    weight_shape = [filter_dimension, filter_dimension, 
                    input_channels, output_channels]
    # attach summary operators to the nodes in the unit
    with tf.name_scope(name):
        with tf.name_scope('filters'):
            filters = weight_variable(weight_shape)
            variable_summaries(filters)
            
            if input_channels <= 4: # if weights makes sense to be visualised
                with tf.name_scope('filter_summary'):
                    filter_grid = put_kernels_on_grid(filters)
                    tf.summary.image('filter_imgs', filter_grid)
            
        with tf.name_scope('biases'):
            biases = bias_variable(output_channels)
            variable_summaries(biases)
        
        # outputs a 4d tensor, shape: [?, input_dim, input_dim, out_chnls]
        layer_output = tf.nn.conv2d(X, filters,
                                    strides = [1,1,1,1], # one at a time
                                    padding = 'VALID') + biases
        
        if with_pooling:
            # applies 2x2 max_pooling to simplify the image
            layer_output = tf.nn.max_pool(layer_output,
                                          ksize = [1,2,2,1],
                                          strides = [1,2,2,1],
                                          padding = 'VALID')
            
        # Rectifier activation
        layer_output = tf.nn.relu(layer_output)
        tf.summary.histogram('activations', layer_output)
        
        return layer_output

def fc_layer(X, num_inputs, num_nodes, 
             use_relu = False,
             with_dropout = False,
             keep_prob = 0.5,
             name = None):
    '''
    Returns the output of ``X`` after going through the fully connected
    layer.
    ``num_inputs`` is the number of inputs that is passed into this layer
    '''
    with tf.name_scope(name):
        # the output of each nodes will be along the 2nd dimension (rows), i.e.
        # the ith datapoint
        with tf.name_scope('weights'):
            weights = weight_variable([num_inputs, num_nodes])
            variable_summaries(weights)
        with tf.name_scope('biases'):
            biases = bias_variable(num_nodes)
            variable_summaries(biases)
        
        z = tf.matmul(X, weights) + biases
        
        # activation
        if use_relu:
            a = tf.nn.relu(z)
        else:
            a = z
            
        tf.summary.histogram('activations', a)
            
        # regularisation, if keep_prob is provided
        if with_dropout and keep_prob is not None:
            return tf.nn.dropout(a, keep_prob)
        else:
            return a