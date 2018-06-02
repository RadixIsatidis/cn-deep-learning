import unittest
import numpy as np
from src.neural_input import normalize, one_hot_encode, neural_net_image_input, neural_net_label_input, \
    neural_net_keep_prob_input


class TestNeuralInput(unittest.TestCase):

    def test_normalize(self):
        test_shape = (np.random.choice(range(1000)), 32, 32, 3)
        test_numbers = np.random.choice(range(256), test_shape)
        normalize_out = normalize(test_numbers)

        self.assertEqual(type(normalize_out).__module__,
                         np.__name__, 'Not Numpy Object')

        self.assertEqual(normalize_out.shape, test_shape,
                         'Incorrect Shape. {} shape found'.format(normalize_out.shape))

        self.assertTrue(normalize_out.max() <= 1 and normalize_out.min() >= 0,
                        'Incorect Range. {} to {} found'.format(normalize_out.min(), normalize_out.max()))

    def test_one_hot_encode(self):
        test_shape = np.random.choice(range(1000))
        test_numbers = np.random.choice(range(10), test_shape)
        one_hot_out = one_hot_encode(test_numbers)

        self.assertEqual(type(one_hot_out).__module__, np.__name__, 'Not Numpy Object')

        self.assertEqual(one_hot_out.shape, (test_shape, 10),
                         'Incorrect Shape. {} shape found'.format(one_hot_out.shape))

        n_encode_tests = 5
        test_pairs = list(zip(test_numbers, one_hot_out))
        test_indices = np.random.choice(len(test_numbers), n_encode_tests)
        labels = [test_pairs[test_i][0] for test_i in test_indices]
        enc_labels = np.array([test_pairs[test_i][1] for test_i in test_indices])
        new_enc_labels = one_hot_encode(labels)

        self.assertTrue(np.array_equal(enc_labels, new_enc_labels),
                        'Encodings returned different results for the same numbers.\n '
                        'For the first call it returned:\n'
                        '{}\n'
                        'For the second call it returned\n'
                        '{}\n'
                        'Make sure you save the map of labels to encodings outside of the function.'
                        .format(enc_labels, new_enc_labels))

    def test_neural_image_input(self):
        image_shape = (32, 32, 3)
        nn_inputs_out_x = neural_net_image_input(image_shape)

        self.assertEquals(nn_inputs_out_x.get_shape().as_list(),
                          [None, image_shape[0], image_shape[1], image_shape[2]],
                          'Incorrect Image Shape.  Found {} shape'.format(nn_inputs_out_x.get_shape().as_list()))

        self.assertEquals(nn_inputs_out_x.op.type, 'Placeholder',
                          'Incorrect Image Type.  Found {} type'.format(nn_inputs_out_x.op.type))

        self.assertEquals(nn_inputs_out_x.name, 'x:0',
                          'Incorrect Name.  Found {}'.format(nn_inputs_out_x.name))

    def test_nn_label_inputs(self):
        n_classes = 10
        nn_inputs_out_y = neural_net_label_input(n_classes)

        self.assertEquals(nn_inputs_out_y.get_shape().as_list(), [None, n_classes],
                          'Incorrect Label Shape.  Found {} shape'.format(nn_inputs_out_y.get_shape().as_list()))

        self.assertEquals(nn_inputs_out_y.op.type, 'Placeholder',
                          'Incorrect Label Type.  Found {} type'.format(nn_inputs_out_y.op.type))

        self.assertEquals(nn_inputs_out_y.name, 'y:0',
                          'Incorrect Name.  Found {}'.format(nn_inputs_out_y.name))

    def test_nn_keep_prob_inputs(self):
        nn_inputs_out_k = neural_net_keep_prob_input()

        self.assertIsNone(nn_inputs_out_k.get_shape().ndims,
                          'Too many dimensions found for keep prob.  Found {} dimensions.  It should be a scalar (0-Dimension Tensor).'.format(
                              nn_inputs_out_k.get_shape().ndims))

        self.assertEquals(nn_inputs_out_k.op.type, 'Placeholder',
                          'Incorrect keep prob Type.  Found {} type'.format(nn_inputs_out_k.op.type))

        self.assertEquals(nn_inputs_out_k.name, 'keep_prob:0',
                          'Incorrect Name.  Found {}'.format(nn_inputs_out_k.name))
