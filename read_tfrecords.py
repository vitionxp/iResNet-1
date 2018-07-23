import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import augmentation

IMAGE_HEIGH = 540
IMAGE_WITCH = 960


def read_data(num_epochs, batch_size, option=True):
    if option:
        data_path = tf.train.match_filenames_once('./tfrecords/train.tfrecords-*')
    else:
        data_path = './tfrecords/test.tfrecords'

    feature = {'left': tf.FixedLenFeature([], tf.string),
               'right': tf.FixedLenFeature([], tf.string),
               'disparity': tf.FixedLenFeature([], tf.string)}

    # Create a list of filenames and pass it to a queue
    filename_queue = tf.train.string_input_producer(data_path, num_epochs=num_epochs, shuffle=True)

    # Define a reader and read the next record
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    # Decode the record read by the reader
    features = tf.parse_single_example(serialized_example, features=feature)

    # Convert the image data from string back to the numbers
    left_img = tf.decode_raw(features['left'], tf.uint8)
    right_img = tf.decode_raw(features['right'], tf.uint8)
    disparity_img = tf.decode_raw(features['disparity'], tf.uint8)

    # Reshape image data into the original shape
    image2 = tf.reshape(left_img, [IMAGE_HEIGH, IMAGE_WITCH, 3])
    image3 = tf.reshape(right_img, [IMAGE_HEIGH, IMAGE_WITCH, 3])
    label = tf.reshape(disparity_img, [IMAGE_HEIGH, IMAGE_WITCH, 1])

    image2, image3, label = augmentation.preprocess_for_train(image2, image3, label, 320, 512, is_training=option)
    return tf.train.shuffle_batch([image2, image3, label], batch_size=batch_size, capacity=1000, num_threads=1,
                                  min_after_dequeue=1)


if __name__ == '__main__':
    with tf.Session() as sess:
        images2, images3, labels = read_data(num_epochs=1, batch_size=2, option=True)
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)

        # Create a coordinator and run all QueueRunner objects
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        try:
            step = 0
            while not coord.should_stop():
                img2, img3, lbl = sess.run([images2, images3, labels])
                plt.figure('dis-lable')
                plt.subplot(311)
                plt.imshow(img2[0])
                plt.subplot(312)
                plt.imshow(img3[0])
                plt.subplot(313)
                plt.imshow(np.reshape(lbl[0], [320, 512]), cmap='gray')

                plt.show()

        except tf.errors.OutOfRangeError:
            print('Done training for  epochs, %d steps.')
        finally:
            coord.request_stop()
        coord.join(threads)
        sess.close()

