import unittest
import tensorflow as tf
from src.con_pool import conv2d_maxpool, flatten, fully_conn, output


class TestConPool(unittest.TestCase):

    def test_conv2d_maxpool(self):
        test_x = tf.placeholder(tf.float32, [None, 32, 32, 5])
        test_num_outputs = 10
        test_con_k = (2, 2)
        test_con_s = (4, 4)
        test_pool_k = (2, 2)
        test_pool_s = (2, 2)

        conv2d_maxpool_out = conv2d_maxpool(test_x, test_num_outputs, test_con_k, test_con_s, test_pool_k, test_pool_s)

        self.assertEquals(conv2d_maxpool_out.get_shape().as_list(), [None, 4, 4, 10],
                          'Incorrect Shape.  Found {} shape'.format(conv2d_maxpool_out.get_shape().as_list()))

    def test_flatten(self):
        test_x = tf.placeholder(tf.float32, [None, 10, 30, 6])
        flat_out = flatten(test_x)

        self.assertEquals(flat_out.get_shape().as_list(), [None, 10 * 30 * 6],
                          'Incorrect Shape.  Found {} shape'.format(flat_out.get_shape().as_list()))

    def test_fully_conn(self):
        test_x = tf.placeholder(tf.float32, [None, 128])
        test_num_outputs = 40

        fc_out = fully_conn(test_x, test_num_outputs)

        self.assertEquals(fc_out.get_shape().as_list(), [None, 40],
                          'Incorrect Shape.  Found {} shape'.format(fc_out.get_shape().as_list()))

    def test_output(self):
        test_x = tf.placeholder(tf.float32, [None, 128])
        test_num_outputs = 40

        output_out = output(test_x, test_num_outputs)

        self.assertEquals(output_out.get_shape().as_list(), [None, 40],
                          'Incorrect Shape.  Found {} shape'.format(output_out.get_shape().as_list()))
