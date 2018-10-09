# 2DSingleObjectLocalization
**Objective:** Localization of rings in synthecic images using a convolutional neural network (CNN). With localization we mean to predict the center of the circle in normalized coordinates relative to the center of the image. 

The left image shows the synthetically generated input into the CNN and the right image shows the prediction (purple rectangle) and the ground truth (green rectangle) for the center of the red ellipse. 

![alt text](https://github.com/gerritschoe/2DSingleObjectLocalization/blob/master/data/7.png "Input") -> ![alt text](https://github.com/gerritschoe/2DSingleObjectLocalization/blob/e11cbce7fb32a05510c5109a5198e8f6cdb123ef/test_output/7.png "Prediction")

We are using PIL (Python Imaging Library) to generate and modify the images and TensorFlow to construct and train the neural network. 

Structure of the neural net that performs regression: 
- input_layer = tf.reshape(features["x"], [-1, 200, 300, 3])
- conv1 = tf.layers.conv2d(inputs=input_layer, filters=4, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
- pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
- conv2 = tf.layers.conv2d(inputs=pool1, filters=8, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
- pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
- pool2_flat = tf.reshape(pool2, [-1, 75 * 50 * 8])
- dropout = tf.layers.dropout(inputs=pool2_flat, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
- dense1 = tf.layers.dense(inputs=dropout, units=200, activation=tf.nn.relu)
- dense2 = tf.layers.dense(inputs=dense1, units=20, activation=tf.nn.relu)
- dense3 = tf.layers.dense(inputs=dense2, units=2, activation=None)
- predictions = {"predict_results": tf.identity(dense3, name="final_layer")

**Computation time:** 
I ran this code on a laptop witht a quadcore CPU, a mid class mobile GPU (2GB VRAM) and 16 GB RAM. 
Time for 100 training steps: 51.750 sec
Visually acceptable predictions are achieved after 2000 steps, good predictions are achieved after 6000 steps. 
A _mean_squared_error_ of 0.01 is reachable. 

**Results:**
The current model shows good convergence and good prediction accuracy. 
Still, there is a lot of room for improvement, for example by switching to advanced localization algorithms like YOLOv3.
A convolutional neural network is not able to achieve outstanding accuracy in regression tasks. 

## Usuage: 

There are 2 independent functionalities: 

1. Generating labeled data in the form of images which contain a single ellipse on a noisy background.
For this, run the file **generateData.py**

2. Train a CNN that predicts the center of the ellipse. 
For this, run **trainCNN_tf.py**

After a defined number of training steps, an evaluation with unseen test data follows. The predictions for the test data are visualized and saved to the _test_output/_ folder. The weights of the neural network are also saved to a defined folder and used as initialization for the next training, if available.

Author: Gerrit Schoettler
Contact: gerrit.schoettler[at]tuhh.de
