import tensorflow as tf
import matplotlib.pyplot as plt
import augmentation
import numpy as np

def main(arg=None):
    with tf.Session() as sess:
        image_1 = tf.gfile.FastGFile("./data/test/left/0400.png", 'rb').read()
        image_2 = tf.gfile.FastGFile("./data/test/right/0400.png", 'rb').read()
        image_3 = tf.gfile.FastGFile("./data/test/disparity/0400.png", 'rb').read()
        with tf.Session() as sess:
            img_data1 = tf.image.decode_png(image_1)
            img_data2 = tf.image.decode_png(image_2)
            img_data3 = tf.image.decode_png(image_3)
            #img_data1 = tf.cast(img_data1, tf.float32)
            img1, img2, img3 = augmentation.preprocess_for_train(img_data1, img_data2, img_data3, 320, 512, True)

            img1 = img1.eval().astype(np.uint8)
            print(img1)
           # img1, img2, img3 = preprocess_for_train(img_data1, img_data2, img_data3, 320, 512)

            plt.subplot(311)
            plt.imshow(img1)
            plt.subplot(312)
            plt.imshow(img2.eval())
            plt.subplot(313)
            plt.imshow(np.reshape(img3.eval(), [320, 512]), cmap='gray')
            plt.show()


if __name__ == '__main__':
    tf.app.run()
