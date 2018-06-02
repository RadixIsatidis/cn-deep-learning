import unittest
import tensorflow as tf
from src.conv_net import conv_net


class TestConvNet(unittest.TestCase):

    def test_conv_net(self):

        test_x = tf.placeholder(tf.float32, [None, 32, 32, 3])
        test_k = tf.placeholder(tf.float32)

        logits_out = conv_net(test_x, test_k)

        assert logits_out.get_shape().as_list() == [None, 10], \
            'Incorrect Model Output.  Found {}'.format(logits_out.get_shape().as_list())
