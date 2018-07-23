from random import shuffle
import glob
import cv2
import numpy as np
import tensorflow as tf
import sys
import os

IMAGE_HEIGH = 540
IMAGE_WITCH = 960


# A function to Load images
def load_image(addr):
    # cv2 load images as BGR, convert it to RGB
    img = cv2.imread(addr)
    img = cv2.resize(img, (IMAGE_WITCH, IMAGE_HEIGH), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.uint8)
    return img


def load_image_gray(addr):
    # cv2 load images as BGR, convert it to RGB
    img = cv2.imread(addr, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMAGE_WITCH, IMAGE_HEIGH), interpolation=cv2.INTER_CUBIC)
    img = img.astype(np.uint8)
    return img


# Convert data to features
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def train_tfrecords(path, choose=True):
    if choose:
        final_path = path + '/train'
        train_filename = './tfrecords/train.tfrecords'
    else:
        final_path = path + '/test'
        train_filename = './tfrecords/test.tfrecords'

    left_path = glob.glob(final_path + '/left/*.png')
    left_path.sort()
    right_path = glob.glob(final_path + '/right/*.png')
    right_path.sort()
    disparity_path = glob.glob(final_path + '/disparity/*.png')
    disparity_path.sort()

    # open the TFRecords file
    writer = tf.python_io.TFRecordWriter(train_filename)

    for i in range(len(left_path)):
        img2 = load_image(left_path[i])
        img3 = load_image(right_path[i])
        label = load_image_gray(disparity_path[i])

        # Create a feature
        feature = {'left': _bytes_feature(tf.compat.as_bytes(img2.tostring())),
                   'right': _bytes_feature(tf.compat.as_bytes(img3.tostring())),
                   'disparity': _bytes_feature(tf.compat.as_bytes(label.tostring()))}

        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        # Serialize to string and write on the file
        writer.write(example.SerializeToString())

    writer.close()
    sys.stdout.flush()


if __name__ == '__main__':
    root_path = os.getcwd() + '/data'
    train_tfrecords(root_path, False)
