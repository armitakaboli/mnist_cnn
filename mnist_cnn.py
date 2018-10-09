import numpy as np
import tensorflow as tf
import os
import sys
tf.app.flags.DEFINE_integer('training_iteration', 30000,
                            'number of training iterations.')
tf.app.flags.DEFINE_integer('model_version', 1, 'version number of the model.')
tf.app.flags.DEFINE_string('work_dir', '/tmp', 'Working directory.')
FLAGS = tf.app.flags.FLAGS
tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(features, labels, mode):
  """CNN model"""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # MNIST images are 28x28 pixels, and have one color channel
  input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])
  # Convolutional Layer #1
  # 32 features using a 3x3 filter with ReLU activation.
  # Output Tensor Shape: [batch_size, 28, 28, 32]
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 28, 28, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 32]
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2_1
  # Computes 64 features using a 3x3 filter.
  # Input Tensor Shape: [batch_size, 14, 14, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 64]
  conv2_1 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #2_1
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 14, 14, 64]
  # Output Tensor Shape: [batch_size, 7, 7, 64]
  pool2_1 = tf.layers.max_pooling2d(inputs=conv2_1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2_2
  # Computes 64 features using a 3x3 filter.
  # Input Tensor Shape: [batch_size, 14, 14, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 64]
  conv2_2 = tf.layers.conv2d(
    inputs=pool1,
    filters=64,
    kernel_size=[3, 3],
    padding="same",
    activation=tf.nn.relu)

  # Pooling Layer #2_2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 14, 14, 64]
  # Output Tensor Shape: [batch_size, 7, 7, 64]
  pool2_2 = tf.layers.max_pooling2d(inputs=conv2_2, pool_size=[2, 2], strides=2)

  # Convolutional Layer #3_1
  # Computes 64 features using a 3x3 filter.
  # Input Tensor Shape: [batch_size, 7, 7, 64]
  # Output Tensor Shape: [batch_size, 7, 7, 256]
  conv3_1 = tf.layers.conv2d(
    inputs=pool2_1,
    filters=256,
    kernel_size=[3, 3],
    padding="same",
    activation=tf.nn.relu)

  # Convolutional Layer #2_2
  # Computes 64 features using a 3x3 filter.
  # Input Tensor Shape: [batch_size, 7, 7, 64]
  # Output Tensor Shape: [batch_size, 7, 7, 256]
  conv3_2 = tf.layers.conv2d(
    inputs=pool2_2,
    filters=256,
    kernel_size=[3, 3],
    padding="same",
    activation=tf.nn.relu)

  conv3 = tf.concat([conv3_1, conv3_2], 3)
  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 7, 7, 256+256]
  # Output Tensor Shape: [batch_size, 7 * 7 * 512]
  pool3_flat = tf.reshape(conv3, [-1, 7 * 7 * 512])

  # Dense Layer 1
  # Densely connected layer with 1000 neurons
  # Input Tensor Shape: [batch_size, 7 * 7 * 512]
  # Output Tensor Shape: [batch_size, 1000]
  dense1 = tf.layers.dense(inputs=pool3_flat, units=1000, activation=tf.nn.relu)

  # Dense Layer 2
  # Densely connected layer with 1000 neurons
  # Input Tensor Shape: [batch_size, 1000]
  # Output Tensor Shape: [batch_size, 500]
  dense2 = tf.layers.dense(inputs=dense1, units=500, activation=tf.nn.relu)
  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=dense2, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits layer
  # Input Tensor Shape: [batch_size, 500]
  # Output Tensor Shape: [batch_size, 10]
  logits = tf.layers.dense(inputs=dropout, units=10)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main():

    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_input = mnist.train.images
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    test_data = mnist.test.images
    test_labels = np.asarray(mnist.test.labels, dtype=np.int32)
    mnist_cnn_classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn, model_dir="./mnist_model")
    tensors_to_log = {"probabilities": "softmax_tensor"}
    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)
    # Training
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_input},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True)
    mnist_cnn_classifier.train(
      input_fn=train_input_fn,
      steps=20000,
      hooks=[logging_hook])

    test_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": test_data},
      y=test_labels,
      num_epochs=1,
      shuffle=False)
    eval_results = mnist_cnn_classifier.evaluate(input_fn=test_input_fn)
    print(eval_results)
    freeze_graph("./mnist_model", "softmax_tensor")


def freeze_graph(model_dir, output_node_names):
    """Extract the sub graph defined by the output nodes and convert
    all its variables into constant
    Args:
        model_dir: the root folder containing the checkpoint state file
        output_node_names: a string, containing all the output node's names,
                            comma separated
    """
    if not tf.gfile.Exists(model_dir):
        raise AssertionError(
            "Export directory doesn't exists. Please specify an export "
            "directory: %s" % model_dir)

    if not output_node_names:
        print("You need to supply the name of a node to --output_node_names.")
        return -1

    # We retrieve our checkpoint fullpath
    checkpoint = tf.train.get_checkpoint_state(model_dir)
    input_checkpoint = checkpoint.model_checkpoint_path

    # We precise the file fullname of our freezed graph
    absolute_model_dir = "/".join(input_checkpoint.split('/')[:-1])
    output_graph = absolute_model_dir + "/frozen_model.pb"

    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True

    # We start a session using a temporary fresh Graph
    with tf.Session(graph=tf.Graph()) as sess:
        # We import the meta graph in the current default Graph
        saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

        # We restore the weights
        saver.restore(sess, input_checkpoint)
        # We use a built-in TF helper to export variables to constants
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, # The session is used to retrieve the weights
            tf.get_default_graph().as_graph_def(), # The graph_def is used to retrieve the nodes
            output_node_names.split(",") # The output node names are used to select the usefull nodes
        )

        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))

    return output_graph_def

main()
