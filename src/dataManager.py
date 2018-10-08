import numpy as np
import pandas as pd
import imageio


RELATIVE_PATH_TO_DATA_FOLDER = '../data/'
labels_csv_name = '_labels.csv'
np.random.seed(1337) # set seed for testing purposes
num_of_folds = 5 # train/test ratio = (num_of_folds - 1)

def load_cvs():
    with open(RELATIVE_PATH_TO_DATA_FOLDER + labels_csv_name, 'rb') as csvfile:
        labels_df = pd.read_csv(csvfile)
        print('Successfully loaded %s datapoints from lables CSV' %labels_df.shape[0])
    return labels_df

def generate_shuffled_indices(num_of_data, num_of_folds):
    indices = np.arange(num_of_data)
    np.random.shuffle(indices)
    print(indices)
    test_set_size = int(num_of_data/num_of_folds)
    train_indices = indices[:(num_of_data - test_set_size)]
    test_indices = indices[(num_of_data - test_set_size):]
    return train_indices, test_indices

def load_images(imageIDs):
    imshape = imageio.imread(RELATIVE_PATH_TO_DATA_FOLDER + str(imageIDs[0]) + '.png').shape
    print((len(imageIDs),list(imshape)))
    images_array = np.zeros((len(imageIDs),imshape[0],imshape[1],imshape[2]))
    #imageID = str(imageIDs)
    for i in range(len(imageIDs)):
        im = imageio.imread(RELATIVE_PATH_TO_DATA_FOLDER + str(imageIDs[i]) + '.png')
        print(im.shape)
    return images_array

def load_train_set():
    pass
def load_test_set():
    pass
labels_df = load_cvs()
train_indices, test_indices = generate_shuffled_indices(labels_df.shape[0], num_of_folds)
print(train_indices)
print(test_indices)
load_images([1, 2])
