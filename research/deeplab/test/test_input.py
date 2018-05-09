from deeplab.input_preprocess import add_location
import numpy as np
import tensorflow as tf


class TestInput(tf.test.TestCase):
    def test_position_matrix(self):
        """
        Creates an image and check the matrix of position
        """
        image = tf.zeros((4,5,3))
        location = add_location(image)

        results = np.zeros((4,5,5))
        for y in range(4):
            for x in range(5):
                results[y, x, 3] = x/5.
                results[y, x, 4] = y/4.

        with self.test_session():
            self.assertAllClose(location.eval(), results)


if __name__ == '__main__':
    tf.test.main()