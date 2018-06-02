import tensorflow as tf
import numpy as np


def conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides):
    """
    Apply convolution then max pooling to x_tensor
    :param x_tensor: TensorFlow Tensor
    :param conv_num_outputs: Number of outputs for the convolutional layer
    :param conv_ksize: kernal size 2-D Tuple for the convolutional layer
    :param conv_strides: Stride 2-D Tuple for convolution
    :param pool_ksize: kernal size 2-D Tuple for pool
    :param pool_strides: Stride 2-D Tuple for pool
    : return: A tensor that represents convolution and max pooling of x_tensor
    """
    depth_input = x_tensor.get_shape().as_list()[3]
    w_conv = tf.Variable(tf.truncated_normal([conv_ksize[0], conv_ksize[1], depth_input, conv_num_outputs],
                                             stddev=.05))
    bias_conv = tf.Variable(tf.zeros(conv_num_outputs))

    x_tensor = tf.nn.conv2d(x_tensor, w_conv,
                            strides=[1, conv_strides[0], conv_strides[1], 1],
                            padding='SAME')
    x_tensor = tf.nn.bias_add(x_tensor, bias_conv)
    x_tensor = tf.nn.relu(x_tensor)

    x_tensor = tf.nn.max_pool(x_tensor,
                              ksize=[1, pool_ksize[0], pool_ksize[1], 1],
                              strides=[1, pool_strides[0], pool_strides[1], 1],
                              padding='SAME')
    return x_tensor


def flatten(x_tensor):
    """
    Flatten x_tensor to (Batch Size, Flattened Image Size)
    : x_tensor: A tensor of size (Batch Size, ...), where ... are the image dimensions.
    : return: A tensor of size (Batch Size, Flattened Image Size).
    """
    x_shape = x_tensor.get_shape().as_list()
    batch_size = tf.shape(x_tensor)[0]  # -1
    return tf.reshape(x_tensor, [batch_size, x_shape[1] * x_shape[2] * x_shape[3]])


def fully_conn(x_tensor, num_outputs):
    """
    Apply a fully connected layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    flattened_shape = np.array(x_tensor.get_shape().as_list()[1:]).prod()
    w_fu = tf.Variable(tf.truncated_normal([flattened_shape, num_outputs],
                                           stddev=.05))
    bias_fu = tf.Variable(tf.zeros(num_outputs))

    x_tensor = tf.add(tf.matmul(x_tensor, w_fu), bias_fu)
    return tf.nn.relu(x_tensor)


def output(x_tensor, num_outputs):
    """
    Apply a output layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    x_shape = x_tensor.get_shape().as_list()
    w_out = tf.Variable(tf.truncated_normal([x_shape[1], num_outputs],
                                            stddev=.05))
    bias_out = tf.Variable(tf.zeros(num_outputs))
    return tf.add(tf.matmul(x_tensor, w_out), bias_out)
