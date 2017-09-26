'''
Created on 24 Sep 2017

@author: pingshiyu
'''

import os, glob, pandas as pd, numpy as np

'''
    Given a root directory to the images, the script will traverse the directory
    and its sub-directories. Saving each image as a flattened row-vector, with 
    the label being the directory name.
    
    Input dir is ``image_master_dir``
    Output file name is specified in ``output_file_name``.
'''

output_file_name = './data/warped_40x40/warped_data_240k.csv'
input_dir_name = './raw_data/dropnone/'

def dir_to_csv(input_dir_name, output_file_name):
    '''
    Takes a root dir saving images with labels as sub-dir's names, saves the
    pictures as flattened row-vectors in ``output_file_name``.
    '''
    # Get all the directories to traverse in ``input_dir_name``
    dir_names = sorted(os.listdir(input_dir_name))
    dir_paths = list(map(lambda dirname: os.path.join(input_dir_name, dirname),
                         dir_names))
    
    # traverse through the directories.
    all_data = np.vstack([_to_numpy_array(label, dir_path) 
                          for label, dir_path in enumerate(dir_paths)])
    
    print('Shuffling data...')
    np.random.shuffle(all_data)
    print('Data shuffled!')
    
    print('Saving data to', output_file_name)
    np.savetxt(output_file_name, all_data, 
               fmt='%d', delimiter=',')
    print('Data successfully saved!')
        
def _to_numpy_array(label, dir_path):
    '''
    Given a ``dir_path`` containing the images of a particular label, return a 
    np array with the first columns being the features, and the last column 
    being the label.
    '''
    print('Processing label', label)
    all_files = glob.glob(os.path.join(dir_path, '*.csv'))
    
    # the dataframe now contains the data
    data_arr = np.stack([pd.read_csv(f, header = None).values.flatten()
                         for f in all_files])
    
    # attach the labels column with labels
    nrows = data_arr.shape[0]
    data_arr = np.column_stack(
        (data_arr, np.full((nrows, 1), label))
        )
    
    return data_arr
        
if __name__ == '__main__':
    dir_to_csv(input_dir_name, output_file_name)
    
    '''
    a = np.array([[1, 2, 3],
                  [4, 5, 6]])
    b = np.array([[7, 8, 9],
                  [10, 11, 12]])
    arr = np.vstack([a,b])
    np.savetxt(output_file_name, arr, 
               fmt='%d', delimiter=',')
    '''