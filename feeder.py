'''
Created on 13 Sep 2017

@author: pingshiyu
'''

import pandas as pd, numpy as np
from PIL import Image

'''
    Feeder object which will provide interface to batch, get the testing set,
    validation set etc - methods necessary to train a neural network. The input
    data will in a csv file, with each datapoint represented in rows. The last 
    entry in each row will be the data's class.
    
    Note that ``one_hot`` option for this class does not currently work - it's 
    off always
'''

class Feeder():
    def __init__(self, 
                 file_path, 
                 shuffle = True, 
                 classes = 24,
                 test_validation_size = 10000,
                 _chunksize = 10000,
                 one_hot = False):
        # Loading data:
        self._filepath = file_path
        self._chunksize = _chunksize
        self._test_validation_size = test_validation_size
        self._reload_data()
        self._make_test_validation_set(test_validation_size)
        
        # number of classes in dataset
        self.classes = classes
        self._shuffle = shuffle
        # contains the current chunk of, by default, 1000
        self._chunk = self._next_chunk()
        # current index on chunk
        self._chunk_index = 0
        # should we use one_hot encoding?
        self.one_hot = one_hot
        
    def _next_chunk(self):
        '''
        Returns next chunk of formatted data, which is a list of
        (data, one_hot_label)
        '''
        # check if iterator has next_chunk. if not reload it
        next_chunk = next(self._data_iter, pd.DataFrame())
        if next_chunk.empty:
            self._reload_data()
            next_chunk = next(self._data_iter)
            
        # print(next_chunk.shape)
        
        # ``raw_chunk`` an np representation of the csv data
        raw_chunk = next_chunk.values
        if self._shuffle:
            np.random.shuffle(raw_chunk)
            
        # size of current chunk
        self._curr_chunksize = raw_chunk.shape[0]
        
        return raw_chunk
    
    def _reload_data(self):
        '''
        Reloads the csv file to start from the beginning
        '''
        print('Epoch complete, reloading data...')
        self._data_iter = pd.read_csv(self._filepath,
                                      sep = ' ',
                                      chunksize = self._chunksize,
                                      memory_map = True,
                                      skiprows = self._test_validation_size
                                      )
        print('Data reloaded!')
        
    def _make_test_validation_set(self, size):
        '''
        Make the test and validation sets. The total size will be ``size``.
        Each one of self.test, self.train is a tuple, with [0] being features, 
        [1] being labels
        '''
        test_validation_data = pd.read_csv(self._filepath,
                                           sep = ' ',
                                           memory_map = True,
                                           nrows = size).values
        self.test = self._format_data(test_validation_data[:(size//2)])
        self.validation = self._format_data(test_validation_data[(size//2):])
    
    def _format_data(self, raw_data, one_hot = False):
        '''
        Takes in a numpy array of data and returns a tuple of (data, labels)
        Where labels are in either integer or one_hot form.
        '''
        if one_hot:
            labels = self._to_one_hot(raw_data[:, -1])
        else:
            labels = raw_data[:, -1]
        data = raw_data[:, :-1]
        
        return data, labels
    
    def _to_one_hot(self, labels):
        '''
        Converts 1d row of labels to one-hot form
        '''
        one_hot_list = [np.eye(self.classes)[y] for y in labels]
        return np.array(one_hot_list).reshape(-1, self.classes)
    
    def next_batch(self, size = 128):
        '''
        Returns a minibatch of ``size`` datapoints
        '''
        i = self._chunk_index
        
        if (i+size) <= self._curr_chunksize:
            xs, ys = self._format_data(self._chunk[i:(i+size)], self.one_hot)
            self._chunk_index += size
            
        else: # i+size > chunksize
            # grab first part of batch
            xs_1, ys_1 = self._format_data(self._chunk[i:], self.one_hot)
            
            remaining = size - (self._curr_chunksize - i)
            
            # begin the next chunk
            self._chunk = self._next_chunk()
            
            # get the second part of the chunk
            xs_2, ys_2 = self._format_data(self._chunk[0:remaining], self.one_hot)
            
            # update the chunk index
            self._chunk_index = remaining
            
            xs = np.append(xs_1, xs_2, axis = 0)
            ys = np.append(ys_1, ys_2, axis = 0)
        
        return xs, ys
    
if __name__ == '__main__':
    '''
    Testing purposes
    '''
    feeder = Feeder('./data/cleanData_small.csv')
    for i in range(10000):
        xs, ys = feeder.next_batch(500)
        print(i, xs.shape, ys.shape)
    