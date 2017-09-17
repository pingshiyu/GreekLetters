'''
Created on 16 Sep 2017

@author: pingshiyu
'''
from PIL import Image
import numpy as np
from feeder import Feeder
from scipy.misc import toimage

# for debug purposes
def to_2d_image(array):
    '''
    1d flattened array of a square image, converted to image object
    Returns the image object
    '''
    side_length = int(np.sqrt(array.size))
    np_2d_arr = np.reshape(array, (side_length, side_length))
    
    return Image.fromarray(np_2d_arr, 'L')

if __name__ == '__main__':
    data = Feeder('./data/cleanData.csv')
    
    xs, ys = data.next_batch(10)
    for x, y in zip(xs, ys):
        toimage(np.reshape(x, (45,45))).show()
        print(y)
        wait = input('')
    