import tensorflow as tf
import numpy as np
from PIL import Image
import os
import unittest

# To avoid to run on GPU
config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )

# PATH TO DIRECTORY TO CHECK
dataset = "/zfsauton/project/public/deep_clustering/data/datasets/cityscapes/"
tfrecords_filename = os.path.join(dataset, "tfrecord_instances/")

original_images, reconstructed_images = [], []

class TestTfrecords(unittest.TestCase):
    def test_image_tfrecord(self):
        for tfrecord in os.listdir(tfrecords_filename):
            if "train" in tfrecord:
                image_original_path = os.path.join(dataset, "leftImg8bit/", "train/")
                instance_path = os.path.join(dataset, "gtFine/", "train/")
            else:
                image_original_path = os.path.join(dataset, "leftImg8bit/", "test/")
                instance_path = os.path.join(dataset, "gtFine/", "test/")

            record_iterator = tf.python_io.tf_record_iterator(path=os.path.join(tfrecords_filename, tfrecord))

            for string_record in record_iterator:
                example = tf.train.Example()
                example.ParseFromString(string_record)
                name = (example.features.feature['image/filename']
                                            .bytes_list
                                            .value[0])

                height = int(example.features.feature['image/height']
                                            .int64_list
                                            .value[0])
                
                width = int(example.features.feature['image/width']
                                            .int64_list
                                            .value[0])
                
                img_string = (example.features.feature['image/encoded']
                                            .bytes_list
                                            .value[0])
                
                annotation_string = (example.features.feature['image/segmentation/class/encoded']
                                            .bytes_list
                                            .value[0])
                
                with tf.Session(config=config) as sess:
                    # Frecords content
                    img_1d = sess.run(tf.image.decode_png(img_string, channels=3))
                    reconstructed_img = img_1d.reshape((height, width, -1))
                    annotation_1d = sess.run(tf.image.decode_png(annotation_string, channels=1))
                    reconstructed_annotation = annotation_1d.reshape((height, width))
                
                    # Original Data
                    city = name[:name.index('_')]
                    img_path = os.path.join(image_original_path, city, name + "_leftImg8bit.png")
                    annotation_path = os.path.join(instance_path, city, name + "_gtFine_instanceTrainIds.png")
                    img = np.array(Image.open(img_path))
                    annotation = np.array(Image.open(annotation_path))

                    self.assertTrue(np.allclose(reconstructed_img, img), msg="Mismatch images - {}".format(name))
                    self.assertTrue(np.allclose(reconstructed_annotation, annotation), msg="Mismatch instances - {}".format(name))

if __name__ == '__main__':
    unittest.main()
