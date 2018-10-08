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
    test_set_size = int(num_of_data/num_of_folds)
    train_indices = indices[:(num_of_data - test_set_size)]
    test_indices = indices[(num_of_data - test_set_size):]
    return train_indices, test_indices

def load_images(imgIndices, pictureIDs):
    imshape = imageio.imread(RELATIVE_PATH_TO_DATA_FOLDER + str(pictureIDs[imgIndices[0]]) + '.png').shape
    images_array = np.zeros((len(imgIndices),imshape[0],imshape[1],imshape[2]))

    for i in range(len(imgIndices)):
        im = imageio.imread(RELATIVE_PATH_TO_DATA_FOLDER + str(pictureIDs[imgIndices[i]]) + '.png')
        images_array[i,:,:,:] = im
        if i % 1000 == 0 and i > 0:
            print('Loaded next 1000 images')

    return images_array

def load_images_and_labels(indices,labels_df, FeatureNames):
    data = load_images(indices, labels_df['PictureID'])/255   # normalization, max color value is 255
    labels = np.array(labels_df.loc[labels_df['PictureID'][indices],FeatureNames])
    return data, labels

def load_train_and_test_data():
    labels_df = load_cvs()
    train_indices, test_indices = generate_shuffled_indices(labels_df.shape[0], num_of_folds)
    FeatureNames = list(labels_df)[1:]
    train_data, train_labels = load_images_and_labels(train_indices, labels_df, FeatureNames)
    test_data, test_labels = load_images_and_labels(test_indices, labels_df, FeatureNames)

    print('DataManager: DONE')
    return train_data, train_labels, test_data, test_labels

if __name__ == "__main__":
    # Short Example:
    train_data, train_labels, test_data, test_labels = load_train_and_test_data()

    # Detailed Example:

    print('FileLoader | Demo')

    # #-#-# Example to load specific data #-#-#
    print('\nLoading CSV-files')
    labels_df = load_cvs()
    print(list(labels_df))
    train_indices, test_indices = generate_shuffled_indices(labels_df.shape[0], num_of_folds)
    print('train_indices = ', train_indices)
    print('test_indices = ', test_indices)

    # #-#-# Example to load images
    print('\nLoading image-files')
    images_array = load_images(train_indices, labels_df['PictureID'])
    FeatureNames = list(labels_df)[1:]
    train_data, train_labels  = load_images_and_labels(train_indices, labels_df, FeatureNames)
    test_data, test_labels = load_images_and_labels(test_indices, labels_df, FeatureNames)
    print('train_data.shape, train_labels.shape = ', train_data.shape, train_labels.shape)
    print('test_data.shape, test_labels.shape = ', test_data.shape, test_labels.shape)
    print('train_labels = ', train_labels)
    print('test_labels = ', test_labels)
    print('***DONE WITH DATAMANAGER DEMO***')