# 2DSingleObjectLocalization
Objective: Localization of rings in synthecic images using a convolutional neural network (CNN). With localization we mean to predict the center of the circle in normalized coordinates relative to the center of the image. 

The left image shows the synthetically generated input into the CNN and the right image shows the prediction (purple rectangle) and the ground truth (green rectangle) for the center of the red ellipse. 

![alt text](https://github.com/gerritschoe/2DSingleObjectLocalization/blob/master/data/7.png "Input") -> ![alt text](https://github.com/gerritschoe/2DSingleObjectLocalization/blob/master/test_output/7.png "Prediction")

We are using PIL (Python Imaging Library) to generate and modify the images and TensorFlow to construct and train the neural network. 
After training, the weights of the neural network are saved on the machine. 

Contact: gerrit.schoettler[at]tuhh.de
