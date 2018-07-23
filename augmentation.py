import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


def distort_color(image, color_ordering=0, seed=0):
    if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255., seed=seed)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5, seed=seed)
        image = tf.image.random_hue(image, max_delta=0.2, seed=seed)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5, seed=seed)
    if color_ordering == 1:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5, seed=seed)
        image = tf.image.random_hue(image, max_delta=0.2, seed=seed)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5, seed=seed)
        image = tf.image.random_brightness(image, max_delta=32. / 255., seed=seed)
    if color_ordering == 2:
        image = tf.image.random_hue(image, max_delta=0.2, seed=seed)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5, seed=seed)
        image = tf.image.random_brightness(image, max_delta=32. / 255., seed=seed)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5, seed=seed)
    if color_ordering == 3:
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5, seed=seed)
        image = tf.image.random_brightness(image, max_delta=32. / 255., seed=seed)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5, seed=seed)
        image = tf.image.random_hue(image, max_delta=0.2, seed=seed)
    return tf.clip_by_value(image, 0.0, 1.0)


def preprocess_for_train(image1, image2, image3, height, width, is_training=True):

    if image1.dtype != tf.float32 and image2.dtype != tf.float32 and image3.dtype != tf.float32:
        image1 = tf.image.convert_image_dtype(image1, dtype=tf.float32)
        image2 = tf.image.convert_image_dtype(image2, dtype=tf.float32)
        image3 = tf.image.convert_image_dtype(image3, dtype=tf.float32)

    seed1 = np.random.randint(10000)
    distorted_image1 = tf.random_crop(image1, [height, width, 3], seed1)
    distorted_image2 = tf.random_crop(image2, [height, width, 3], seed1)
    distorted_image3 = tf.random_crop(image3, [height, width, 1], seed1)

    seed2 = np.random.randint(10000)
    seed3 = np.random.randint(10000)

    distorted_image1 = tf.image.random_flip_left_right(distorted_image1, seed=seed2)
    distorted_image2 = tf.image.random_flip_left_right(distorted_image2, seed=seed2)
    distorted_image3 = tf.image.random_flip_left_right(distorted_image3, seed=seed2)
    distorted_image1 = tf.image.random_flip_up_down(distorted_image1, seed=seed3)
    distorted_image2 = tf.image.random_flip_up_down(distorted_image2, seed=seed3)
    distorted_image3 = tf.image.random_flip_up_down(distorted_image3, seed=seed3)

    option = np.random.randint(4)
    seed4 = np.random.randint(1000)
    distorted_image1 = distort_color(distorted_image1, option, seed=seed4)
    distorted_image2 = distort_color(distorted_image2, option, seed=seed4)

    distorted_image1 = tf.image.convert_image_dtype(distorted_image1, dtype=tf.uint8)
    distorted_image2 = tf.image.convert_image_dtype(distorted_image2, dtype=tf.uint8)
    distorted_image3 = tf.image.convert_image_dtype(distorted_image3, dtype=tf.uint8)
    distorted_image3.set_shape([height, width, 1])

    return distorted_image1, distorted_image2, distorted_image3


def main(arg=None):
    image_1 = tf.gfile.FastGFile("./data/test/left/0400.png", 'rb').read()
    image_2 = tf.gfile.FastGFile("./data/test/right/0400.png", 'rb').read()
    image_3 = tf.gfile.FastGFile("./data/test/disparity/0400.png", 'rb').read()
    with tf.Session() as sess:
        img_data1 = tf.image.decode_png(image_1)
        img_data2 = tf.image.decode_png(image_2)
        img_data3 = tf.image.decode_png(image_3)
        img1, img2, img3 = preprocess_for_train(img_data1, img_data2, img_data3, 320, 512)
        plt.subplot(311)
        plt.imshow(img1.eval())
        plt.subplot(312)
        plt.imshow(img2.eval())
        plt.subplot(313)
        plt.imshow(np.reshape(img3.eval(), [320, 512]), cmap='gray')
        plt.show()


if __name__ == '__main__':
    tf.app.run()
