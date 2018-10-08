from PIL import Image, ImageDraw
import os

path_up = os.path.dirname(os.path.dirname(__file__))
data_path = '/test_output/'

def visualize_prediction(test_data, test_labels, predicted_labels, test_indices):

    for i in range(test_data.shape[0]):
        img = test_data[i,:,:,:]
        print(img.shape)
