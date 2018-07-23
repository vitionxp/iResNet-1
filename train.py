import net
import tensorflow as tf
import read_tfrecords
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

IMAGE_WITCH = 512
IMAGE_HIGHT = 320
global_step = tf.Variable(0, trainable=False)

with tf.name_scope('Input'):
    left_image = tf.placeholder(tf.float32, [None, IMAGE_HIGHT, IMAGE_WITCH, 3])
    right_image = tf.placeholder(tf.float32, [None, IMAGE_HIGHT, IMAGE_WITCH, 3])
    label_image = tf.placeholder(tf.float32, [None, IMAGE_HIGHT, IMAGE_WITCH, 1])
    disparity = net.net(left_image, right_image)
with tf.name_scope('cost'):
    cost = net.loss(label_image, disparity)
    tf.summary.scalar('cost', cost)
with tf.name_scope('train'):
    learning_rate = tf.train.exponential_decay(0.0001, global_step, 10000, 0.5, staircase=True)
    train = tf.train.AdamOptimizer(learning_rate).minimize(cost, global_step=global_step)

merged = tf.summary.merge_all()

with tf.Session() as sess:
    images2, images3, labels = read_tfrecords.read_data(num_epochs=50, batch_size=2, option=True)
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)

    writer = tf.summary.FileWriter('logs/', sess.graph)
    # Create a coordinator and run all QueueRunner objects
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    try:
        step = 0
        while not coord.should_stop():
            img2, img3, lbl = sess.run([images2, images3, labels])
            _, c, summary, l = sess.run([train, cost, merged, learning_rate], feed_dict={left_image: img2, right_image: img3, label_image: lbl})
            if step % 10 == 1:
                print('Step %d: loss = %.2f learning = %.8f' % (step, c, l))
                writer.add_summary(summary, step)
                saver = tf.train.Saver()
                saver.save(sess, 'checkpoint/my_net.ckpt')
            step += 1

    except tf.errors.OutOfRangeError:
        print('Done training for  epochs, %d steps.')
    finally:
        # Wait for threads to stop
        coord.request_stop()
    saver = tf.train.Saver()
    saver.save(sess, 'checkpoint/my_net.ckpt')
    coord.join(threads)
    sess.close()
