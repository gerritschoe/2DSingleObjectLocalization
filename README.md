# 2DSingleObjectLocalization
Objective: Localization of rings in synthecic images using a convolutional neural network (CNN). With localization we mean to predict the center of the circle in normalized coordinates relative to the center of the image. 

The left image shows the synthetically generated input into the CNN and the right image shows the prediction (purple rectangle) and the ground truth (green rectangle) for the center of the red ellipse. 

![alt text](https://github.com/gerritschoe/2DSingleObjectLocalization/blob/master/data/7.png "Input") -> ![alt text](https://github.com/gerritschoe/2DSingleObjectLocalization/blob/e11cbce7fb32a05510c5109a5198e8f6cdb123ef/test_output/7.png "Prediction")

We are using PIL (Python Imaging Library) to generate and modify the images and TensorFlow to construct and train the neural network. 

## Usuage: 

There are 2 independent functionalities: 

1. Generating labeled data in the form of images containing a single ellipse.
For this, run the file **generateData.py**

2. Train a CNN on this data.
For this, run **trainCNN_tf.py**

After a defined number of training steps, an evaluation with unseen test data follows. The predictions for the test data are visualized and saved to the _test_output/_ folder. The weights of the neural network are also saved to a defined folder and used as initialization for the next training, if available.

This project is currently in progress and changes are expected. 

Author: Gerrit Schoettler
Contact: gerrit.schoettler[at]tuhh.de
