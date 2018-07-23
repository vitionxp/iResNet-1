import tensorflow as tf
import tensorflow.contrib.slim as slim
import math
import correlation
import read_tfrecords
from tensorflow.python.ops import nn
import downsample
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv


IMAGE_WITCH = 512
IMAGE_HIGHT = 320


def net(left, right):
    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], activation_fn=nn.relu, padding='SAME'):
        with tf.name_scope('Multi-Share'):
            conv1a = slim.conv2d(left, 64, [7, 7], stride=2, scope='conv1', reuse=False)
            conv1b = slim.conv2d(right, 64, [7, 7], stride=2, scope='conv1', reuse=True)

            up_1a = slim.conv2d_transpose(conv1a, 32, [4, 4], stride=2, scope='up_1', reuse=False)
            up_1b = slim.conv2d_transpose(conv1b, 32, [4, 4], stride=2, scope='up_1', reuse=True)

            conv2a = slim.conv2d(conv1a, 128, [5, 5], stride=2, scope='conv2', reuse=False)
            conv2b = slim.conv2d(conv1b, 128, [5, 5], stride=2, scope='conv2', reuse=True)

            up_2a = slim.conv2d_transpose(conv2a, 32, [8, 8], stride=4, scope='up_2', reuse=False)
            up_2b = slim.conv2d_transpose(conv2b, 32, [8, 8], stride=4, scope='up_2', reuse=True)

            up_1a2a_concat = tf.concat([up_1a, up_2a], 3)
            up_1b2b_concat = tf.concat([up_1b, up_2b], 3)
            up_1a2a = slim.conv2d(up_1a2a_concat, 32, [1, 1], scope='up_12_concat', reuse=False)
            up_1b2b = slim.conv2d(up_1b2b_concat, 32, [1, 1], scope='up_12_concat', reuse=True)

        with tf.name_scope('DES-net'):
            corr1d = correlation.correlation(conv2a, conv2b, 1, 8, 1, 2, 8)
            conv_redir = slim.conv2d(conv2a, 64, [1, 1])
            corr1d_redir_concat = tf.concat([corr1d, conv_redir], 3)
            conv3 = slim.conv2d(corr1d_redir_concat, 256, [3, 3], stride=2)
            conv3_1 = slim.conv2d(conv3, 256, [3, 3])

            conv4 = slim.conv2d(conv3_1, 512, [3, 3], stride=2)
            conv4_1 = slim.conv2d(conv4, 512, [3, 3])

            conv5 = slim.conv2d(conv4_1, 512, [3, 3], stride=2)
            conv5_1 = slim.conv2d(conv5, 512, [3, 3])

            conv6 = slim.conv2d(conv5, 1024, [3, 3], stride=2)
            conv6_1 = slim.conv2d(conv6, 1024, [3, 3])

            disp6 = slim.conv2d(conv6_1, 1, [3, 3], activation_fn=None)
            resized_6 = tf.image.resize_images(disp6, [int(math.ceil(IMAGE_HIGHT / 32.)), int(math.ceil(IMAGE_WITCH / 32.))])

            up_conv5 = slim.conv2d_transpose(conv6_1, 512, [4, 4], stride=2)
            iconv5_concat = tf.concat([up_conv5, resized_6, conv5_1], 3)
            iconv5 = slim.conv2d(iconv5_concat, 512, [3, 3])

            disp5 = slim.conv2d(iconv5, 1, [3, 3], activation_fn=None)
            resized_5 = tf.image.resize_images(disp5, [int(math.ceil(IMAGE_HIGHT / 16.)), int(math.ceil(IMAGE_WITCH / 16.))])

            up_conv4 = slim.conv2d_transpose(iconv5, 256, [4, 4], stride=2)
            iconv4_concat = tf.concat([up_conv4, resized_5, conv4_1], 3)
            iconv4 = slim.conv2d(iconv4_concat, 256, [3, 3])

            disp4 = slim.conv2d(iconv4, 1, [3, 3], activation_fn=None)
            resized_4 = tf.image.resize_images(disp4, [int(math.ceil(IMAGE_HIGHT / 8.)), int(math.ceil(IMAGE_WITCH / 8.))])

            up_conv3 = slim.conv2d_transpose(iconv4, 128, [4, 4], stride=2)
            iconv3_concat = tf.concat([up_conv3, resized_4, conv3_1], 3)
            iconv3 = slim.conv2d(iconv3_concat, 128, [3, 3])

            disp3 = slim.conv2d(iconv3, 1, [3, 3], activation_fn=None)
            resized_3 = tf.image.resize_images(disp3, [int(math.ceil(IMAGE_HIGHT / 4.)), int(math.ceil(IMAGE_WITCH / 4.))])

            up_conv2 = slim.conv2d_transpose(iconv3, 64, [4, 4], stride=2)
            iconv2_concat = tf.concat([up_conv2, resized_3, conv2a], 3)
            iconv2 = slim.conv2d(iconv2_concat, 64, [3, 3])

            disp2 = slim.conv2d(iconv2, 1, [3, 3], activation_fn=None)
            resized_2 = tf.image.resize_images(disp2, [int(math.ceil(IMAGE_HIGHT / 2.)), int(math.ceil(IMAGE_WITCH / 2.))])

            up_conv1 = slim.conv2d_transpose(iconv2, 32, [4, 4], stride=2)
            iconv1_concat = tf.concat([up_conv1, resized_2, conv1a], 3)
            iconv1 = slim.conv2d(iconv1_concat, 32, [3, 3])

            disp1 = slim.conv2d(iconv1, 1, [3, 3], activation_fn=None)
            resized_1 = tf.image.resize_images(disp1, [IMAGE_HIGHT, IMAGE_WITCH])

            up_conv0 = slim.conv2d_transpose(iconv1, 32, [4, 4], stride=2)
            iconv0_concat = tf.concat([up_conv0, resized_1, up_1a2a], 3)
            iconv0 = slim.conv2d(iconv0_concat, 32, [3, 3])

            disp0 = slim.conv2d(iconv0, 1, [3, 3], activation_fn=None)

        with tf.name_scope('DRS-net'):
            r_conv0_concat = tf.concat([abs(up_1a2a - up_1b2b), disp0, up_1a2a], 3)
            r_conv0 = slim.conv2d(r_conv0_concat, 32, [3, 3])

            r_conv1 = slim.conv2d(r_conv0, 64, [3, 3], stride=2)

            c_conv1a = slim.conv2d(conv1a, 16, [3, 3], scope='c_conv1', reuse=False)
            c_conv1b = slim.conv2d(conv1b, 16, [3, 3], scope='c_conv1', reuse=True)

            r_corr = correlation.correlation(c_conv1a, c_conv1b, 1, 4, 1, 2, 4)

            r_conv1_1_concat = tf.concat([r_conv1, r_corr], 3)
            r_conv1_1 = slim.conv2d(r_conv1_1_concat, 64, [3, 3])

            r_conv2 = slim.conv2d(r_conv1_1, 128, [3, 3], stride=2)
            r_conv2_1 = slim.conv2d(r_conv2, 128, [3, 3])

            r_res2 = slim.conv2d(r_conv2_1, 1, [3, 3], activation_fn=None)
            r_res2_resize = tf.image.resize_images(r_res2, [int(math.ceil(IMAGE_HIGHT / 2)), int(math.ceil(IMAGE_WITCH / 2))])

            r_upconv1 = slim.conv2d_transpose(r_conv2_1, 64, [4, 4], stride=2)
            r_iconv1_concat = tf.concat([r_upconv1, r_res2_resize, r_conv1_1], 3)
            r_iconv1 = slim.conv2d(r_iconv1_concat, 64, [3, 3])

            r_res1 = slim.conv2d(r_iconv1, 1, [3, 3], activation_fn=None)
            r_res1_resize = tf.image.resize_images(r_res1, [IMAGE_HIGHT, IMAGE_WITCH])

            r_upconv0 = slim.conv2d_transpose(r_conv1, 32, [4, 4], stride=2)
            r_iconv0_concat = tf.concat([r_upconv0, r_res1_resize, r_conv0], 3)
            r_iconv0 = slim.conv2d(r_iconv0_concat, 32, [3, 3])

            r_res0 = slim.conv2d(r_iconv0, 1, [3, 3], activation_fn=None)

        return {
            'disp6': disp6,
            'disp5': disp5,
            'disp4': disp4,
            'disp3': disp3,
            'disp2': disp2,
            'disp1': disp1,
            'disp0': disp0 + r_res0
        }


def loss(flow, predictions):
    losses = []
    # L2 loss between predict_disp6, blob23 (weighted w/ 0.32)
    predict_disp6 = predictions['disp6']
    size = [predict_disp6.shape[1], predict_disp6.shape[2]]
    downsampled_disp6 = downsample.downsample(flow, size)
    losses.append(tf.losses.mean_squared_error(downsampled_disp6, predict_disp6))

    # L2 loss between predict_disp5, blob28 (weighted w/ 0.08)
    predict_disp5 = predictions['disp5']
    size = [predict_disp5.shape[1], predict_disp5.shape[2]]
    downsampled_disp5 = downsample.downsample(flow, size)
    losses.append(tf.losses.mean_squared_error(downsampled_disp5, predict_disp5))

    # L2 loss between predict_disp4, blob33 (weighted w/ 0.02)
    predict_disp4 = predictions['disp4']
    size = [predict_disp4.shape[1], predict_disp4.shape[2]]
    downsampled_disp4 = downsample.downsample(flow, size)
    losses.append(tf.losses.mean_squared_error(downsampled_disp4, predict_disp4))

    # L2 loss between predict_disp3, blob38 (weighted w/ 0.01)
    predict_disp3 = predictions['disp3']
    size = [predict_disp3.shape[1], predict_disp3.shape[2]]
    downsampled_disp3 = downsample.downsample(flow, size)
    losses.append(tf.losses.mean_squared_error(downsampled_disp3, predict_disp3))

    # L2 loss between predict_disp2, blob43 (weighted w/ 0.005)
    predict_disp2 = predictions['disp2']
    size = [predict_disp2.shape[1], predict_disp2.shape[2]]
    downsampled_disp2 = downsample.downsample(flow, size)
    losses.append(tf.losses.mean_squared_error(downsampled_disp2, predict_disp2))

    predict_disp1 = predictions['disp1']
    size = [predict_disp1.shape[1], predict_disp1.shape[2]]
    downsampled_disp1 = downsample.downsample(flow, size)
    losses.append(tf.losses.mean_squared_error(downsampled_disp1, predict_disp1))

    predict_disp0 = predictions['disp0']
    size = [predict_disp0.shape[1], predict_disp0.shape[2]]
    downsampled_disp0 = downsample.downsample(flow, size)
    losses.append(tf.losses.mean_squared_error(downsampled_disp0, predict_disp0))

    #loss = tf.losses.compute_weighted_loss(losses, [0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32])
    # Return the 'total' loss: loss fns + regularization terms defined in the model
    return losses[0] * 0.32 + losses[1] * 0.16 + losses[2] * 0.08 + losses[3] * 0.04 + losses[4] * 0.02 + losses[5] * 0.01 + losses[6] * 0.005


if __name__ == '__main__':

    left_image = tf.placeholder(tf.float32, [None, IMAGE_HIGHT, IMAGE_WITCH, 3])
    right_image = tf.placeholder(tf.float32, [None, IMAGE_HIGHT, IMAGE_WITCH, 3])
    label_image = tf.placeholder(tf.float32, [None, IMAGE_HIGHT, IMAGE_WITCH, 1])
    disparity = net(left_image, right_image)

    cost = loss(label_image, disparity)
    train = tf.train.AdamOptimizer(0.0001).minimize(cost)

    with tf.Session() as sess:
        image2, image3, label = read_tfrecords.read_data(num_epochs=10, batch_size=1, option=True)
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        try:
            step = 0
            while not coord.should_stop():
                img2, img3, lbl = sess.run([image2, image3, label])
                _, t = sess.run([train, cost], feed_dict={left_image: img2, right_image: img3, label_image: lbl})
                print(t)
        except tf.errors.OutOfRangeError:
            print('Done training for  epochs, %d steps.')
        finally:
            coord.request_stop()
        coord.join(threads)
        sess.close()


