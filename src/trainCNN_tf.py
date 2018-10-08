# Modified the example code provided on
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/layers/cnn_mnist.py
# Tutorial: https://www.tensorflow.org/tutorials/estimators/cnn
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0" # 0 = GPU on, -1 = GPU off

import numpy as np
import tensorflow as tf

from dataManager import load_train_and_test_data

tf.logging.set_verbosity(tf.logging.INFO)

# Our application logic will be added here
def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    # Our images are 300x200 pixels and have 3 color channels
    # batct_size = -1 indicates dynamic batch size based on the input
    input_layer = tf.reshape(features["x"], [-1, 200, 300, 3])

    # Convolutional Layer #1
    # Computes 32 features using a 5x5 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 300, 200, 3]
    # Output Tensor Shape: [batch_size, 300, 200, 32]
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=4,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    # First max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 300, 200, 32]
    # Output Tensor Shape: [batch_size, 150, 100, 32]
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2
    # Computes 64 features using a 5x5 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 150, 100, 32]
    # Output Tensor Shape: [batch_size, 150, 100, 64]
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=8,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #2
    # Second max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 150, 100, 64]
    # Output Tensor Shape: [batch_size, 75, 50, 64]
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Flatten tensor into a batch of vectors
    # Input Tensor Shape: [batch_size, 75, 50, 64]
    # Output Tensor Shape: [batch_size, 75 * 50 * 64]
    pool2_flat = tf.reshape(pool2, [-1, 75 * 50 * 8])

    # Dense Layers
    # Densely connected layer with 1024 neurons
    # Input Tensor Shape: [batch_size, 75 * 50 * 64]
    # Output Tensor Shape: [batch_size, 1024]
    dense = tf.layers.dense(inputs=pool2_flat, units=500, activation=tf.nn.relu)

    # Add dropout operation; 0.6 probability that element will be kept
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    dense2 = tf.layers.dense(inputs=dropout, units=100, activation=tf.nn.relu)

    # Logits layer
    # Input Tensor Shape: [batch_size, 1024]
    # Output Tensor Shape: [batch_size, 10]
    predictions = tf.layers.dense(inputs=dense2, units=2, activation=None)

    #predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        # "predict_results": dense3
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        #"probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    #}
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.mean_squared_error(labels=labels, predictions=predictions)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "mse": tf.metrics.mean_squared_error(
            labels=labels, predictions=predictions["predict_results"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def main(unused_argv):
    # Load training and eval data

    train_data2, train_labels2, test_data2, test_labels2 = load_train_and_test_data()


    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images  # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    test_data = mnist.test.images  # Returns np.array
    test_labels = np.asarray(mnist.test.labels, dtype=np.int32)
    print('Comparing train_data2, train_labels2, test_data2, test_labels2:')
    print(train_data.shape, train_data2.shape)
    print(train_labels.shape, train_labels2.shape)
    print(test_data.shape, test_data2.shape)
    print(test_labels.shape, test_labels2.shape)
    print(type(train_data), type(train_data2))
    print(type(train_labels), type(train_labels2))
    print(type(test_data), type(test_data2))
    print(type(test_labels), type(test_labels2))

    train_data, train_labels, test_data, test_labels = load_train_and_test_data()

    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model6")

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=500)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True)
    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=2000, #default: 20k
        #hooks=[logging_hook]) # logging hook optional, outputs probability tensors (long print in console)
        )

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": test_data},
        y=test_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)

if __name__ == "__main__":
    #train_data, train_labels, test_data, test_labels = load_train_and_test_data()
    tf.app.run()

  