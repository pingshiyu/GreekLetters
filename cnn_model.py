'''
Created on 16 Sep 2017

@author: pingshiyu
'''

import tensorflow as tf, numpy as np

class Model():
    def __init__(self, model_dir):
        self.sess = tf.Session()
        # load our saved meta-graph
        saver = tf.train.import_meta_graph(model_dir + '/model_final.ckpt.meta')
        # load values into our meta-graph and thus into our session
        saver.restore(self.sess, tf.train.latest_checkpoint(model_dir))
        
        self.graph = tf.get_default_graph()
        # debug 
        # print([n.name for n in self.graph.as_graph_def().node])
        
        # get the placeholders
        self.X = self.graph.get_tensor_by_name('inputs/Placeholder:0')
        self.keep_prob = self.graph.get_tensor_by_name('Placeholder:0')
        self.predictions = self.graph.get_tensor_by_name('accuracy/ArgMax:0')
        self.activations = self.graph.get_tensor_by_name('output/add:0')
        
    def make_prediction(self, X):
        '''
        Make prediction based on X. The output tensor we shall extract is the
        ``predictions`` tensor in our original graph.
        
        Returns the predicted class, along with the confidence of the prediction
        Applies softmax for the probability of the prediction
        '''
        predictions, activations = (self.sess).run([self.predictions, self.activations], 
                                                   feed_dict = {self.X: X, self.keep_prob: 1.0})
        ''' DEBUG
        np.set_printoptions(threshold=np.nan)
        print('activations', activations[0])
        print('probabilities', self.softmax(activations[0]))'''
        prediction = predictions[0]
        confidence = self.softmax(activations[0])[prediction]
        
        return prediction, confidence
        
    def softmax(self, arr):
        '''
        Returns the softmax of ``arr``
        '''
        return np.exp(arr) / np.sum(np.exp(arr), axis = 0)

if __name__ == '__main__':
    a = tf.placeholder(tf.float32)
    c = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    d = tf.constant([[1.0, 1.0], [0.0, 1.0]])
    e = tf.matmul(c, d, name='example')
    
    with tf.Session() as sess:
        test =  sess.run(e)
        print(e.name) #example:0
        test = [n.name for n in tf.get_default_graph().as_graph_def().node]
        print(test) #Tensor("example:0", shape=(2, 2), dtype=float32)