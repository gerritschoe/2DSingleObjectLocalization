from PIL import Image, ImageDraw
import os
import random
import csv
import numpy as np
from addNoise import noisy

NumOfGeneratedPics = 1000
imgMode = 'RGB'; imgSize = [300,200]; imgColor = 'black'

csvLabels = ['PictureID', 'XcenterEllipse', 'YcenterEllipse']

random.seed(1337)

def draw_ellipse(image, bounds, width=1, outline='red', antialias=4):
    """Improved ellipse drawing function, based on PIL.ImageDraw."""
    # credit to Håken Lid, source:
    # https://stackoverflow.com/questions/32504246/draw-ellipse-in-python-pil-with-line-thickness
    # Use a single channel image (mode='L') as mask.
    # The size of the mask can be increased relative to the imput image
    # to get smoother looking results.
    mask = Image.new(
        size=[int(dim * antialias) for dim in image.size],
        mode='L', color='black')
    draw = ImageDraw.Draw(mask)

    # draw outer shape in white (color) and inner shape in black (transparent)
    for offset, fill in (width/-2.0, 'white'), (width/2.0, 'black'):
        left, top = [(value + offset) * antialias for value in bounds[:2]]
        right, bottom = [(value - offset) * antialias for value in bounds[2:]]
        draw.ellipse([left, top, right, bottom], fill=fill)

    # downsample the mask using PIL.Image.LANCZOS
    # (a high-quality downsampling filter).
    mask = mask.resize(image.size, Image.LANCZOS)
    # paste outline color to input image through the mask
    image.paste(outline, mask=mask)

def calcEllipseBox(x, y, XcenterEllipse, YcenterEllipse,XsizeEllipse, YsizeEllipse):
    leftBotX = x / 2 + XcenterEllipse * x / 2 - (XsizeEllipse / 2) * x
    leftBotY = y / 2 - YcenterEllipse * y / 2 - (YsizeEllipse / 2) * x
    rightTopX = x / 2 + XcenterEllipse * x / 2 + (XsizeEllipse / 2) * x
    rightTopY = y / 2 - YcenterEllipse * y / 2 + (YsizeEllipse / 2) * x

    ellipse_box = [leftBotX, leftBotY, rightTopX, rightTopY]  # bottom left and top right corners of the ellipse box
    return ellipse_box

path_up = os.path.dirname(os.path.dirname(__file__))
data_path = '/data/'
cvsFileName = '_labels.csv'
with open(path_up + data_path + cvsFileName, 'w', newline='') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerow(csvLabels)

csvFile.close()

for i in range(NumOfGeneratedPics):
    img = Image.new(mode=imgMode, size=imgSize, color=imgColor)
    x, y = img.size
    Xc = random.uniform(-0.7, 0.7)
    Yc = random.uniform(-0.7, 0.7)
    #XYfactor = random.gauss(0,0.2)
    c = random.uniform(0.1, 0.5)
    d = c + random.uniform(-0.5, 0.5)*c

    [XcenterEllipse, YcenterEllipse] = [Xc, Yc] # [0,0] is center, [1,1] is top right, [-1,-1] is bottom left
    [XsizeEllipse, YsizeEllipse] = [c, d] # positive values, size of the ellipse relative to the size of the image

    ellipse_box = calcEllipseBox(x, y, XcenterEllipse, YcenterEllipse,XsizeEllipse, YsizeEllipse)

    # draw a thick white ellipse and a thin black ellipse
    draw_ellipse(img, ellipse_box, outline='red', width=10, antialias=4)
    imgArray = np.array(img)
    imgArrayNoisy = noisy('gauss', imgArray)
    imgNoisy = Image.fromarray(np.uint8(imgArrayNoisy))

    my_file = '' + str(i) #'ellipse' + str(i)

    imgNoisy.save(path_up + data_path + my_file + '.png', encoding='utf-8')
    img.close()
    imgNoisy.close()

    # generate label file
    row = [i, Xc, Yc]
    with open(path_up + data_path + cvsFileName, 'a', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(row)

    csvFile.close()

print('Generated %s images and the _lable.cvs file in data/' % (str(NumOfGeneratedPics)))




