import net
import tensorflow as tf
import read_tfrecords
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

IMAGE_WITCH = 512
IMAGE_HIGHT = 320


with tf.name_scope('Input'):
    left_image = tf.placeholder(tf.float32, [None, IMAGE_HIGHT, IMAGE_WITCH, 3])
    right_image = tf.placeholder(tf.float32, [None, IMAGE_HIGHT, IMAGE_WITCH, 3])
    label_image = tf.placeholder(tf.float32, [None, IMAGE_HIGHT, IMAGE_WITCH, 1])
    disparity = net.net(left_image, right_image)

with tf.name_scope('cost'):
    cost = net.loss(label_image, disparity)

with tf.Session() as sess:
    images2, images3, labels = read_tfrecords.read_data(num_epochs=1, batch_size=1, option=True)
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    try:
        step = 0
        saver = tf.train.Saver()
        saver.restore(sess, 'checkpoint/my_net.ckpt')

        while not coord.should_stop():
            img2, img3, lbl = sess.run([images2, images3, labels])
            dis, c = sess.run([disparity['disp0'], cost], feed_dict={left_image: img2, right_image: img3, label_image: lbl})
            if step % 1 == 0:
                print('Step %d: loss = %.2f ' % (step, c))
                # writer.add_summary(summery, step)
                if step < 50:
                    dis = dis.astype(np.uint8)
                    plt.figure('image-disparity-prediction')

                    plt.subplot(221)
                    plt.imshow(img2[0])

                    plt.subplot(222)
                    plt.imshow(img3[0])

                    plt.subplot(223)
                    plt.imshow(np.reshape(lbl[0], [320, 512]), cmap='gray')

                    plt.subplot(224)
                    plt.imshow(np.reshape(dis[0], [320, 512]), cmap='gray')

                    plt.show()
            step += 1

    except tf.errors.OutOfRangeError:
        print('Done training for  epochs, %d steps.')
    finally:
        # Wait for threads to stop
        coord.request_stop()
    coord.join(threads)
    sess.close()



