import numpy as np
import tensorflow as tf


tf.logging.set_verbosity(tf.logging.INFO)


def cnn_model_fn(features, labels, mode):
    input_layer = tf.reshape(features, [-1, 28, 28, 1])

    # input is an array of size 100 * 28 * 28
    # 100 is the batch size
    #
    # [b ,W, H, c] = [-1, 28, 28, 1] -> shape
    #
    # [
    #     [ [c], [c], [c], [c], [c], ......28 times ],
    #     [ [c], [c], [c], [c], [c], ......28 times ],
    #     [ [c], [c], [c], [c], [c], ......28 times ],
    #     [ [c], [c], [c], [c], [c], ......28 times ],
    # ..... 28 times ]
    #
    # if c is 3(R, G, B),
    # c = [R, G, B]
    #
    # b = -1 divides the whole data set to arrays of shape [ W, H, c ]
    # which gives master array of length -> input.length / W * H * c
    #
    #


    # Convolutional layer 1 with Relu activation

    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)


    # Pooling layer 1
    pool1 = tf.layers.max_pooling2d(conv1, pool_size=[2,2], strides=2)


    # Convolutional Layer 2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)


    # Pooling layer 2
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)


    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
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


################################### Same as Regression and Classification ###################################

# Fetch MNIST data set
mnist = tf.contrib.learn.datasets.load_dataset("mnist")

# Split to train and test data
train_data = mnist.train.images # Returns np.array
train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
eval_data = mnist.test.images # Returns np.array
eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)


# create input functions
train_input_function = tf.estimator.inputs.numpy_input_fn(train_data, y=train_labels,
                                                          batch_size=100, num_epochs=None,
                                                          shuffle=True)

eval_input_fn = tf.estimator.inputs.numpy_input_fn(
                eval_data,
                y=eval_labels,
                num_epochs=1,
                shuffle=False)



# Create the Model - use the above function
mnist_classifier = tf.estimator.Estimator(
  model_fn=cnn_model_fn, model_dir="models/mnist_convnet_model")


# Setting up logs
tensors_to_log = {"probabilities": "softmax_tensor"}

logging_hook = tf.train.LoggingTensorHook(
  tensors=tensors_to_log, every_n_iter=50)


# Training the model
mnist_classifier.train(train_input_function,
                       steps=20000,
                       hooks=[logging_hook])


# Evaluating the model
eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
print(eval_results)
