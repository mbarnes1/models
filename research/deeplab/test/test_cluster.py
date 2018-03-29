from deeplab.utils.cluster_utils import kwik_cluster, lp_cost
from functools import partial
import numpy as np
import unittest


class MyTestCase(unittest.TestCase):
    def setUp(self):
        pass

    def test_perfect_singular_vectors(self):
        """
        Clustering should be perfect.
        """
        singular_vectors = np.array([[1, 0, 0], [1., 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]]).astype(np.float)
        cost = partial(lp_cost, p=1)
        labels = kwik_cluster(singular_vectors, cost)
        self.assertEqual(labels[0], labels[1])
        self.assertEqual(labels[2], labels[3])
        self.assertEqual(labels[4], labels[5])
        self.assertNotEqual(labels[1], labels[2])
        self.assertNotEqual(labels[3], labels[4])
        self.assertNotEqual(labels[1], labels[4])

    def test_l2_cost(self):
        x = np.array([1.0, 0.9, 0.5, 0.0])
        costs = lp_cost(x, 2)
        np.testing.assert_almost_equal(costs, np.array([1.0, 0.81/0.82, 0.5, 0]))