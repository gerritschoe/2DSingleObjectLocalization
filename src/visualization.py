import os
import numpy as np
from PIL import Image, ImageDraw

path_up = os.path.dirname(os.path.dirname(__file__))
output_path = '/test_output/'
point_size = 0.03

def visualize_prediction(test_data, test_labels, predicted_labels, test_indices):

    for i in range(test_data.shape[0]):
        img = Image.fromarray(np.uint8(test_data[i,:,:,:]*255))
        x, y = img.size
        draw = ImageDraw.Draw(img)

        XcenterEllipse, YcenterEllipse, XsizeEllipse, YsizeEllipse = calcRectangleSize(test_labels[i])
        ellipse_box = calcEllipseBox(x, y, XcenterEllipse, YcenterEllipse, XsizeEllipse, YsizeEllipse)
        draw.rectangle(ellipse_box, fill = 'green', outline ='white')

        XcenterEllipse, YcenterEllipse, XsizeEllipse, YsizeEllipse = calcRectangleSize(predicted_labels[i])
        ellipse_box = calcEllipseBox(x, y, XcenterEllipse, YcenterEllipse, XsizeEllipse, YsizeEllipse)
        draw.rectangle(ellipse_box, fill='purple', outline='white')

        #img.show()     # the images will pop up after evaluation

        my_file = '' + str(test_indices[i])
        img.save(path_up + output_path + my_file + '.png', encoding='utf-8')
        img.close()

def calcRectangleSize(lable):
    [XcenterEllipse, YcenterEllipse] = lable  # [0,0] is center, [1,1] is top right, [-1,-1] is bottom left
    [XsizeEllipse, YsizeEllipse] = [point_size, point_size] # positive values, size of the ellipse relative to the size of the image
    return XcenterEllipse, YcenterEllipse, XsizeEllipse, YsizeEllipse

# function from generateData.py
def calcEllipseBox(x, y, XcenterEllipse, YcenterEllipse,XsizeEllipse, YsizeEllipse):
    leftBotX = x / 2 + XcenterEllipse * x / 2 - (XsizeEllipse / 2) * x
    leftBotY = y / 2 - YcenterEllipse * y / 2 - (YsizeEllipse / 2) * x
    rightTopX = x / 2 + XcenterEllipse * x / 2 + (XsizeEllipse / 2) * x
    rightTopY = y / 2 - YcenterEllipse * y / 2 + (YsizeEllipse / 2) * x

    ellipse_box = [leftBotX, leftBotY, rightTopX, rightTopY]  # bottom left and top right corners of the ellipse box
    return ellipse_box