# Modified the example code provided on
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/layers/cnn_mnist.py
# Tutorial: https://www.tensorflow.org/tutorials/estimators/cnn
# Author: Gerrit Schoettler, Email: gerrit.schoettler[at]tuhh.de
# Created 09.10.2018

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0" # 0 = GPU on, -1 = GPU off

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from dataManager import load_train_and_test_data
from visualization import visualize_prediction

tf.logging.set_verbosity(tf.logging.INFO)   # prints training progress every 100 steps

# Our application logic will be added here
def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    # Our images are 300x200 pixels and have 3 color channels
    # batct_size = -1 indicates dynamic batch size based on the input
    input_layer = tf.reshape(features["x"], [-1, 200, 300, 3])

    # Convolutional Layer #1: Computes 8 features, 5x5 filter and ReLU activation.
    # Padding means zeros are added outside the image to preserve width and height.
    # In: [batch_size, 300, 200, 3], Out: [batch_size, 300, 200, 8]
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=4,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1: First max pooling layer with a 2x2 filter and stride of 2
    # In: [batch_size, 300, 200, 8], Out: [batch_size, 150, 100, 8]
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2: 16 features with 5x5 filter, padding on.
    # In: [batch_size, 150, 100, 8], Out: [batch_size, 150, 100, 16]
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=8,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #2: 2x2 filter, stide of 2.
    # In: [batch_size, 75, 50, 8], Out: [batch_size, 75, 50, 8]
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Flatten tensor into a batch of vectors. Has no effect on network.
    # In: [batch_size, 75, 50, 8], Out: [batch_size, 75 * 50 * 8]
    pool2_flat = tf.reshape(pool2, [-1, 75 * 50 * 8])

    # Dense Layer #1: Densely connected layer with 100 neurons
    # In: [batch_size, 75 * 50 * 64], Out [batch_size, 100]
    dense1 = tf.layers.dense(inputs=pool2_flat, units=100, activation=tf.nn.relu)

    # Add dropout operation: 0.6 probability that a weight will not be changed during training
    dropout = tf.layers.dropout(
        inputs=dense1, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Dense Layer #2: Densely connected layer with 100 neurons
    # In: [batch_size, 100], Out [batch_size, 100]
    dense2 = tf.layers.dense(inputs=dropout, units=20, activation=tf.nn.relu)

    # Final Layer: No activation (linar layer) needed for regression. ReLU would not allow negative values.
    # In: [batch_size, 100], Out: [batch_size, 2]
    dense3 = tf.layers.dense(inputs=dense2, units=2, activation=None)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "predict_results": tf.identity(dense3, name="final_layer")
        #"probabilities": tf.nn.l2_loss(dense3, name="softmax_tensor")  # softmax not useful in regression
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions["predict_results"],)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.mean_squared_error(labels=labels, predictions=predictions["predict_results"])

    loss_summary = {
        # Generate predictions (for PREDICT and EVAL mode)
        "current_loss": tf.identity(loss, name="current_loss")
        # "probabilities": tf.nn.l2_loss(dense3, name="softmax_tensor")  # softmax not useful in regression
    }

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "mean_squared_error": tf.metrics.mean_squared_error(
            labels=labels, predictions=predictions["predict_results"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def main(unused_argv):
    # Load training and eval data
    train_data, train_labels, test_data, test_labels, test_indices = load_train_and_test_data()

    # Create the Estimator
    regressionCNN = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="../model/convnet_model"
    )

    # Set up logging for predictions
    tensors_to_log = {"current_loss": "current_loss"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log,
        every_n_iter=500
    )

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=5,
        num_epochs=None,
        shuffle=True)
    regressionCNN.train(
        input_fn=train_input_fn,
        steps=1, #default: 20k
        #hooks=[logging_hook] # logging hook optional, outputs probability tensors (long print in console)
        )

    # turn off GPU for evaluation due to lack of enough VRAM
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 0 = GPU on, -1 = GPU off

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": test_data},
        y=test_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = regressionCNN.evaluate(input_fn=eval_input_fn)

    visualization_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": test_data[0:50,:]},
        y=test_labels[0:50,:],
        num_epochs=1,
        shuffle=False)
    results = regressionCNN.predict(input_fn=visualization_input_fn)

    n_viz = 50
    predicted_labels = np.zeros_like(test_labels[0:n_viz,:])
    j = 0
    for result in results:
        predicted_labels[j,:] = result
        j = j+1

    print("\nPREDICTION OF TEST DATA FINISHED")
    print(eval_results)

    print("\n First 3 predicted_labels: \n",  predicted_labels[0:3,:])
    print("First 3 test_labels: \n", test_labels[0:3,:])

    visualize_prediction(test_data[0:n_viz,:], test_labels[0:n_viz,:], predicted_labels, test_indices[0:n_viz])

    print("GENERATED %s prediction images in test_output/" %n_viz)
if __name__ == "__main__":
    tf.app.run()

  