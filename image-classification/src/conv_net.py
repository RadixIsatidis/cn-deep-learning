import tensorflow as tf
from src.con_pool import conv2d_maxpool, flatten, fully_conn, output


def conv_net(x, keep_prob, init_conv_num_outputs=32, output_fully=1024):
    """
    Create a convolutional neural network model
    : x: Placeholder tensor that holds image data.
    : keep_prob: Placeholder tensor that hold dropout keep probability.
    : return: Tensor that represents logits
    """
    # TODO: Apply 1, 2, or 3 Convolution and Max Pool layers
    #    Play around with different number of outputs, kernel size and stride
    # Function Definition from Above:
    #    conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides)
    for idx in range(1, 4):
        conv_num_outputs = idx * init_conv_num_outputs
        x = conv2d_maxpool(x, conv_num_outputs, (5, 5), (1, 1), (2, 2), (2, 2))

    # TODO: Apply a Flatten Layer
    # Function Definition from Above:
    #   flatten(x_tensor)
    x = flatten(x)

    # TODO: Apply 1, 2, or 3 Fully Connected Layers
    #    Play around with different number of outputs
    # Function Definition from Above:
    #   fully_conn(x_tensor, num_outputs)
    for idx in range(1, 4):
        x = fully_conn(x, output_fully)
        x = tf.nn.dropout(x, keep_prob)

    # TODO: Apply an Output Layer
    #    Set this to the number of classes
    # Function Definition from Above:
    #   output(x_tensor, num_outputs)
    x = output(x, 10)

    # TODO: return output
    return x
