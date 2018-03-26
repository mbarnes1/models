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
            loss = spectral_loss(labels_tensor, embeddings, instance_mask, subsample_power=8)
            self.assertAlmostEqual(loss.eval(), 0.)

            # Induce 1 FP and 1 FN by flipping one label
            # 2*2/5^2 = 0.16
            labels = [[0, 0, 1, 2, 1]]
            labels_tensor = tf.convert_to_tensor(labels)  # 1 x 5
            loss = spectral_loss(labels_tensor, embeddings, instance_mask, subsample_power=12)
            self.assertAlmostEqual(loss.eval(), 0.16, places=1)
