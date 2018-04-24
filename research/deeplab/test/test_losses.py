from __future__ import division
from deeplab.utils.losses import batch_gather, _tile_along_new_axis, labels_to_adjacency, spectral_loss
import numpy as np
from scipy.spatial.distance import pdist, squareform
import tensorflow as tf


class MyTestCase(tf.test.TestCase):

    def test_tile_along_new_axis(self):
        with self.test_session():
            x = [1, 2, 3, 4]
            x_tensor = tf.convert_to_tensor(x)
            x_tiled = _tile_along_new_axis(x_tensor, 2, axis=0)
            self.assertAllEqual(x_tiled.eval(), [x, x])

    def test_batch_gather_2d(self):
        with self.test_session():
            batch_size, n = 2, 5
            A = tf.reshape(tf.range(0, batch_size*n), [batch_size, n])  # batch_size x n
            indices = tf.convert_to_tensor([[0, 3], [1, 4]])  # batch_size x 2
            B = batch_gather(A, indices)
            self.assertAllEqual(B.eval(), [[0, 3], [6, 9]])

    def test_batch_gather_3d(self):
        with self.test_session():
            batch_size, n, m = 2, 5, 3
            A = np.reshape(np.arange(0, batch_size * n * m), [batch_size, n, m])
            A_tensor = tf.convert_to_tensor(A)  # batch_size x n x k
            indices = [[0, 3], [1, 4]]
            indices_tensor = tf.convert_to_tensor(indices)  # batch_size x 2
            B_tensor = batch_gather(A_tensor, indices_tensor)  # batch_size x 2 x m
            B_array = [[A[0, indices[0][0], :], A[0, indices[0][1], :]],
                       [A[1, indices[1][0], :], A[1, indices[1][1], :]]]
            self.assertAllEqual(B_tensor.eval(), B_array)

    def test_labels_to_adjacency(self):
        with self.test_session():
            labels = np.array([[0, 0, 1, 1, 2, 2]])
            A = pdist(labels.transpose(), metric='hamming')
            A = [~squareform(A).astype('bool')]
            labels_tensor = tf.convert_to_tensor(labels)
            A_tensor = labels_to_adjacency(labels_tensor)
            self.assertAllEqual(A_tensor.eval(), A)

    def test_spectral_loss(self):
        with self.test_session():
            labels = [[0, 0, 1, 2, 2]]
            labels_tensor = tf.convert_to_tensor(labels)  # 1 x 5
            embeddings = tf.one_hot(labels_tensor, 3)  # 1 x 5 x 3
            instance_mask = tf.ones((1, 5))
            loss = spectral_loss(labels_tensor, embeddings, instance_mask, subsample_power=8, rebalance_classes=False)
            self.assertAlmostEqual(loss.eval(), 0.)

            # Induce 1 FP and 1 FN by flipping one label
            # 2*2/5^2 = 0.16
            labels = [[0, 0, 1, 2, 1]]
            labels_tensor = tf.convert_to_tensor(labels)  # 1 x 5
            loss = spectral_loss(labels_tensor, embeddings, instance_mask, subsample_power=12, rebalance_classes=False)
            self.assertAlmostEqual(loss.eval(), 0.16, places=1)

    def test_spectral_loss_semantic(self):
        with self.test_session():
            labels = [[0000, 0000, 1000, 2000, 2000]]
            labels_tensor = tf.convert_to_tensor(labels)  # 1 x 5
            embeddings = tf.one_hot(labels_tensor, 2002)  # 1 x 5 x 2001
            instance_mask = tf.ones((1, 5))
            loss = spectral_loss(labels_tensor, embeddings, instance_mask, subsample_power=None,
                                 no_semantic_blocking=False, rebalance_classes=False)
            self.assertAlmostEqual(loss.eval(), 0.)

            # 1 false positive instance
            # 9 total (within semantic class) edges (both directions)
            # 2 incorrect
            labels = [[0000, 0000, 1000, 2001, 2000]]
            labels_tensor = tf.convert_to_tensor(labels)  # 1 x 5
            loss = spectral_loss(labels_tensor, embeddings, instance_mask, subsample_power=None,
                                 no_semantic_blocking=False, rebalance_classes=False)
            self.assertAlmostEqual(loss.eval(), 2.0/9.0)

    def test_spectral_loss_semantic_2d(self):
        with self.test_session():
            labels = [[0000, 0000, 1000, 2000, 2000], [0000, 1000, 1000, 0000, 0000]]
            labels_tensor = tf.convert_to_tensor(labels)  # 2 x 5
            embeddings = tf.one_hot(labels_tensor, 2002)  # 2 x 5 x 2001
            instance_mask = tf.ones((2, 5))
            loss = spectral_loss(labels_tensor, embeddings, instance_mask, subsample_power=8,
                                 no_semantic_blocking=False, rebalance_classes=False)
            self.assertAlmostEqual(loss.eval(), 0.)

    def test_rebalance_classes(self):
        with self.test_session():

            # Semantic class 2 has 2 FPs and 2 TPs = error 0.5
            # Total of 3 classes, so 0.5/3 net error.
            pred_labels = [[0000, 0000, 1000, 2000, 2000]]
            pred_labels_tensor = tf.convert_to_tensor(pred_labels)  # 1 x 5
            embeddings = tf.one_hot(pred_labels_tensor, 2002)  # 1 x 5 x 2001

            true_labels = [[0000, 0000, 1000, 2001, 2000]]
            true_labels_tensor = tf.convert_to_tensor(true_labels)  # 1 x 5
            instance_mask = tf.ones((1, 5))
            loss = spectral_loss(true_labels_tensor, embeddings, instance_mask, subsample_power=14,
                                 no_semantic_blocking=False, rebalance_classes=True)
            self.assertAlmostEqual(loss.eval(), 1/3*1/2, places=2)


if __name__ == '__main__':
    tf.test.main()
