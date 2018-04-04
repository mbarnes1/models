from deeplab.round_embeddings import eval_embedding
from deeplab.utils.cluster_utils import kwik_cluster, lp_cost
from functools import partial
import imageio
import numpy as np
import os
import unittest


class TestClusterUtils(unittest.TestCase):
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


class TestEmbeddings(unittest.TestCase):
    def setUp(self):
        pass

    def test_eval_embedding(self):
        embedding_path = 'data/embeddings/frankfurt_000000_000294.npy'
        semantic_path = 'data/gtFine/val/frankfurt/frankfurt_000000_000294_gtFine_labelIds.png'
        gt_instance_path = 'data/gtFine/val/frankfurt/frankfurt_000000_000294_gtFine_instanceIds.png'
        image_name = 'frankfurt_000000_000294'
        results_dir = 'data/roundings/'

        make_perfect_embedding(gt_instance_path, embedding_path)

        results_dict = eval_embedding(embedding_path, semantic_path, gt_instance_path, results_dir, image_name)
        self.assertEqual(results_dict['averages']['allAp'], 1.0)


def make_perfect_embedding(input_file_path, output_file_path):
    """
    Create a perform hot-one embedding using the true instance labels.
    :param input_file_path: Path to PNG file of instance labels
    :param output_file_path: Path to write embedding npy file
    """
    instance_labels = imageio.imread(input_file_path)
    unique_labels = np.unique(instance_labels)
    element_map = {unique_labels[i]: i for i in range(0, len(unique_labels))}
    new_instance_labels = np.copy(instance_labels)
    for k, v in element_map.iteritems(): new_instance_labels[instance_labels == k] = v
    embedding = np.eye(len(unique_labels))[new_instance_labels]
    if not os.path.exists(os.path.dirname(output_file_path)):
        os.makedirs(os.path.dirname(output_file_path))
    np.save(output_file_path, embedding)