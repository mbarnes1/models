from __future__ import division
from deeplab.model import _add_location
import numpy as np
import tensorflow as tf


class TestInput(tf.test.TestCase):
    def test_position_matrix(self):
        """
        Creates an image and check the matrix of position
        """
        batch = 2
        height = 4
        width = 5
        channels = 3
        image = tf.zeros((batch, height, width, channels))
        location = _add_location(image)

        position_x = np.arange(0., 1., 1./width)
        position_y = np.arange(0., 1., 1./height)

        with self.test_session():
            self.assertAllClose(location.eval()[0, 0, :, -2], position_x)
            self.assertAllClose(location.eval()[0, :, 0, -1], position_y)


if __name__ == '__main__':
    tf.test.main()