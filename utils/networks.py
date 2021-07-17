from __future__ import print_function
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tf_slim as slim
from tf_slim.layers import utils

import numpy as np
import sys
from datetime import datetime
import os
from PIL import Image
import six
import math
import imageio


def crop_features(feature, out_size):
    """Crop the center of a feature map
    Args:
    feature: Feature map to crop
    out_size: Size of the output feature map
    Returns:
    Tensor that performs the cropping
    """
    # Crop the center of a feature map
    up_size = tf.shape(feature)
    ini_w = tf.div(tf.subtract(up_size[1], out_size[1]), 2)
    ini_h = tf.div(tf.subtract(up_size[2], out_size[2]), 2)
    slice_input = tf.slice(feature, (0, ini_w, ini_h, 0), (-1, out_size[1], out_size[2], -1))
    return tf.reshape(slice_input, [int(feature.get_shape()[0]), out_size[1], out_size[2], int(feature.get_shape()[3])])


def middle_flow_block(inpt, num_outputs=728, kernel_size=None, unit_num=None):
    if kernel_size is None:
        kernel_size = [3, 3]
    unit_num = str(unit_num)

    residual = inpt
    net = tf.nn.relu(inpt)
    net = slim.separable_conv2d(net, num_outputs, kernel_size,
                                scope='xception_65/middle_flow/block1/unit_{}/xception_module/separable_conv1_depthwise'.format(
                                    unit_num))
    net = slim.batch_norm(net,
                          scope='xception_65/middle_flow/block1/unit_{}/xception_module/separable_conv1_pointwise/BatchNorm'.format(
                              unit_num))
    net = tf.nn.relu(net)



    net = slim.separable_conv2d(net, num_outputs, kernel_size,
                                scope='xception_65/middle_flow/block1/unit_{}/xception_module/separable_conv2_depthwise'.format(
                                    unit_num))
    net = slim.batch_norm(net,
                          scope='xception_65/middle_flow/block1/unit_{}/xception_module/separable_conv2_pointwise/BatchNorm'.format(
                              unit_num))
    net = tf.nn.relu(net)



    net = slim.separable_conv2d(net, num_outputs, kernel_size,
                                scope='xception_65/middle_flow/block1/unit_{}/xception_module/separable_conv3_depthwise'.format(
                                    unit_num))
    net = slim.batch_norm(net,
                          scope='xception_65/middle_flow/block1/unit_{}/xception_module/separable_conv3_pointwise/BatchNorm'.format(
                              unit_num))
    residual_next = tf.math.add(net, residual)

    return residual_next


#  TODO: figure out if the transpose convolution can be trained and how it works
def xfcn(inputs, dropout_rate, scope='xfcn'):
    """Defines the xfcn network
    Args:
    inputs: Tensorflow placeholder that contains the input image
    scope: Scope name for the network
    Returns:
    net: Output Tensor of the network
    end_points: Dictionary with all Tensors of the network
    """
    im_size = tf.shape(inputs)

    with tf.variable_scope(scope, 'xfcn', [inputs]) as sc:
        end_points_collection = sc.name + '_end_points'
        # Collect outputs of all intermediate layers.
        with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                            outputs_collections=end_points_collection):
            # Entry flow
            # Block 1
            net = slim.conv2d(inputs, 32, [3, 3], stride=2, padding='VALID', scope='xception_65/entry_flow/conv1_1')
            net = slim.batch_norm(net, scope='xception_65/entry_flow/conv1_1/BatchNorm')
            net = tf.nn.relu(net)
            net = slim.conv2d(net, 64, [3, 3], scope='xception_65/entry_flow/conv1_2')
            net = slim.batch_norm(net, scope='xception_65/entry_flow/conv1_2/BatchNorm')
            net = tf.nn.relu(net)
            residual_1 = slim.conv2d(net, 128, [1, 1], stride=2,
                                     scope='xception_65/entry_flow/block1/unit_1/xception_module/shortcut')
            residual_1 = slim.batch_norm(residual_1,
                                         scope='xception_65/entry_flow/block1/unit_1/xception_module/shortcut/BatchNorm')

            # block 2
            net = slim.separable_conv2d(net, 128, [3, 3], activation_fn=None,
                                        scope='xception_65/entry_flow/block1/unit_1/xception_module/separable_conv1_depthwise')
            net = slim.batch_norm(net,
                                  scope='xception_65/entry_flow/block1/unit_1/xception_module/separable_conv1_pointwise/BatchNorm')

            net = tf.nn.relu(net)



            net = slim.separable_conv2d(net, 128, [3, 3],
                                        scope='xception_65/entry_flow/block1/unit_1/xception_module/separable_conv2_depthwise')
            net = slim.batch_norm(net,
                                  scope='xception_65/entry_flow/block1/unit_1/xception_module/separable_conv2_pointwise/BatchNorm')
            net = tf.nn.relu(net)



            net = slim.separable_conv2d(net, 128, [3, 3],
                                        scope='xception_65/entry_flow/block1/unit_1/xception_module/separable_conv3_depthwise')
            net = slim.batch_norm(net,
                                  scope='xception_65/entry_flow/block1/unit_1/xception_module/separable_conv3_pointwise/BatchNorm')

            net = slim.max_pool2d(net, [3, 3], stride=2, padding='SAME')

            net_2 = tf.math.add(residual_1, net)

            net_2_drop = slim.dropout(net_2, keep_prob=dropout_rate)

            residual_2 = slim.conv2d(net_2, 256, [1, 1], stride=2,
                                     scope='xception_65/entry_flow/block2/unit_1/xception_module/shortcut')
            residual_2 = slim.batch_norm(residual_2,
                                         scope='xception_65/entry_flow/block2/unit_1/xception_module/shortcut/BatchNorm')

            # block 3
            net = tf.nn.relu(net_2)
            net = slim.separable_conv2d(net, 256, [3, 3],
                                        scope='xception_65/entry_flow/block2/unit_1/xception_module/separable_conv1_depthwise')
            net = slim.batch_norm(net,
                                  scope='xception_65/entry_flow/block2/unit_1/xception_module/separable_conv1_pointwise/BatchNorm')
            net = tf.nn.relu(net)



            net = slim.separable_conv2d(net, 256, [3, 3],
                                        scope='xception_65/entry_flow/block2/unit_1/xception_module/separable_conv2_depthwise')
            net = slim.batch_norm(net,
                                  scope='xception_65/entry_flow/block2/unit_1/xception_module/separable_conv2_pointwise/BatchNorm')
            net = tf.nn.relu(net)



            net = slim.separable_conv2d(net, 256, [3, 3],
                                        scope='xception_65/entry_flow/block2/unit_1/xception_module/separable_conv3_depthwise')
            net = slim.batch_norm(net,
                                  scope='xception_65/entry_flow/block2/unit_1/xception_module/separable_conv3_pointwise/BatchNorm')

            net = slim.max_pool2d(net, [3, 3], stride=2, padding='SAME')
            net_3 = tf.math.add(net, residual_2)

            net_3_drop = slim.dropout(net_3, keep_prob=dropout_rate)

            residual_3 = slim.conv2d(net_3, 728, [1, 1], stride=2,
                                     scope='xception_65/entry_flow/block3/unit_1/xception_module/shortcut')
            residual_3 = slim.batch_norm(residual_3,
                                         scope='xception_65/entry_flow/block3/unit_1/xception_module/shortcut/BatchNorm')

            # block 4
            net = tf.nn.relu(net_3)
            net = slim.separable_conv2d(net, 728, [3, 3],
                                        scope='xception_65/entry_flow/block3/unit_1/xception_module/separable_conv1_depthwise')
            net = slim.batch_norm(net,
                                  scope='xception_65/entry_flow/block3/unit_1/xception_module/separable_conv1_pointwise/BatchNorm')
            net = tf.nn.relu(net)



            net = slim.separable_conv2d(net, 728, [3, 3],
                                        scope='xception_65/entry_flow/block3/unit_1/xception_module/separable_conv2_depthwise')
            net = slim.batch_norm(net,
                                  scope='xception_65/entry_flow/block3/unit_1/xception_module/separable_conv2_pointwise/BatchNorm')
            net = tf.nn.relu(net)



            net = slim.separable_conv2d(net, 728, [3, 3],
                                        scope='xception_65/entry_flow/block3/unit_1/xception_module/separable_conv3_depthwise')
            net = slim.batch_norm(net,
                                  scope='xception_65/entry_flow/block3/unit_1/xception_module/separable_conv3_pointwise/BatchNorm')

            net = slim.max_pool2d(net, [3, 3], stride=2, padding='SAME')
            net_4 = tf.math.add(net, residual_3)

            net_4_drop = slim.dropout(net_4, keep_prob=dropout_rate)

            # middle flow
            # block 5
            net = middle_flow_block(net_4, unit_num=1)
            # block 6 - 20
            net = middle_flow_block(net, unit_num=2)
            net_5_drop = slim.dropout(net, keep_prob=dropout_rate)

            # Exit flow
            residual_20 = slim.conv2d(net, 1024, [1, 1], stride=2,
                                      scope='xception_65/exit_flow/block1/unit_1/xception_module/shortcut')
            residual_20 = slim.batch_norm(residual_20,
                                          scope='xception_65/exit_flow/block1/unit_1/xception_module/shortcut/BatchNorm')
            # block 21
            net = tf.nn.relu(net)
            net = slim.separable_conv2d(net, 728, [3, 3],
                                        scope='xception_65/exit_flow/block1/unit_1/xception_module/separable_conv1_depthwise')
            net = slim.batch_norm(net,
                                  scope='xception_65/exit_flow/block1/unit_1/xception_module/separable_conv1_pointwise/BatchNorm')
            net = tf.nn.relu(net)



            net = slim.separable_conv2d(net, 1024, [3, 3],
                                        scope='xception_65/exit_flow/block1/unit_1/xception_module/separable_conv2_depthwise')
            net = slim.batch_norm(net,
                                  scope='xception_65/exit_flow/block1/unit_1/xception_module/separable_conv2_pointwise/BatchNorm')
            net = tf.nn.relu(net)



            net = slim.separable_conv2d(net, 1024, [3, 3],
                                        scope='xception_65/exit_flow/block1/unit_1/xception_module/separable_conv3_depthwise')
            net = slim.batch_norm(net,
                                  scope='xception_65/exit_flow/block1/unit_1/xception_module/separable_conv3_pointwise/BatchNorm')
            net = slim.max_pool2d(net, [3, 3], stride=2, padding='SAME')
            net_6 = tf.math.add(net, residual_20)


            net_6_drop = slim.dropout(net_6, keep_prob=dropout_rate)

            # Get side outputs of the network
            with slim.arg_scope([slim.conv2d], biases_initializer=tf.zeros_initializer()):
                side_2 = slim.conv2d(net_2_drop, 16, [3, 3], rate=1, scope='conv2_2_16')

                side_3 = slim.conv2d(net_3_drop, 16, [3, 3], rate=2, scope='conv3_3_16')

                side_4 = slim.conv2d(net_4_drop, 16, [3, 3], rate=4, scope='conv4_3_16')

                side_5 = slim.conv2d(net_5_drop, 16, [3, 3], rate=4, scope='conv5_3_16')

                side_6 = slim.conv2d(net_6_drop, 16, [3, 3], rate=8, scope='conv6_3_16')

                # Supervise side outputs
                side_2_s = slim.conv2d(side_2, 1, [1, 1], scope='score-dsn_2')
                side_3_s = slim.conv2d(side_3, 1, [1, 1], scope='score-dsn_3')
                side_4_s = slim.conv2d(side_4, 1, [1, 1], scope='score-dsn_4')
                side_5_s = slim.conv2d(side_5, 1, [1, 1], scope='score-dsn_5')
                side_6_s = slim.conv2d(side_6, 1, [1, 1], scope='score-dsn_6')
                with slim.arg_scope([slim.convolution2d_transpose],
                                    outputs_collections=end_points_collection):
                    # Side outputs
                    side_2_s = slim.convolution2d_transpose(side_2_s, 1, 8, 4, scope='score-dsn_2-up')
                    side_2_s = crop_features(side_2_s, im_size)
                    utils.collect_named_outputs(end_points_collection, 'xfcn/score-dsn_2-cr', side_2_s)

                    side_3_s = slim.convolution2d_transpose(side_3_s, 1, 16, 8, scope='score-dsn_3-up')
                    side_3_s = crop_features(side_3_s, im_size)
                    utils.collect_named_outputs(end_points_collection, 'xfcn/score-dsn_3-cr', side_3_s)

                    side_4_s = slim.convolution2d_transpose(side_4_s, 1, 32, 16, scope='score-dsn_4-up')
                    side_4_s = crop_features(side_4_s, im_size)
                    utils.collect_named_outputs(end_points_collection, 'xfcn/score-dsn_4-cr', side_4_s)

                    side_5_s = slim.convolution2d_transpose(side_5_s, 1, 32, 16, scope='score-dsn_5-up')
                    side_5_s = crop_features(side_5_s, im_size)
                    utils.collect_named_outputs(end_points_collection, 'xfcn/score-dsn_5-cr', side_5_s)

                    side_6_s = slim.convolution2d_transpose(side_6_s, 1, 64, 32, scope='score-dsn_6-up')
                    side_6_s = crop_features(side_6_s, im_size)
                    utils.collect_named_outputs(end_points_collection, 'xfcn/score-dsn_6-cr', side_6_s)

                    # Main output
                    side_2_f = slim.convolution2d_transpose(side_2, 16, 8, 4, scope='score-multi2-up')
                    side_2_f = crop_features(side_2_f, im_size)
                    utils.collect_named_outputs(end_points_collection, 'xfcn/side-multi2-cr', side_2_f)

                    side_3_f = slim.convolution2d_transpose(side_3, 16, 16, 8, scope='score-multi3-up')
                    side_3_f = crop_features(side_3_f, im_size)
                    utils.collect_named_outputs(end_points_collection, 'xfcn/side-multi3-cr', side_3_f)

                    side_4_f = slim.convolution2d_transpose(side_4, 16, 32, 16, scope='score-multi4-up')
                    side_4_f = crop_features(side_4_f, im_size)
                    utils.collect_named_outputs(end_points_collection, 'xfcn/side-multi4-cr', side_4_f)

                    side_5_f = slim.convolution2d_transpose(side_5, 16, 32, 16, scope='score-multi5-up')
                    side_5_f = crop_features(side_5_f, im_size)
                    utils.collect_named_outputs(end_points_collection, 'xfcn/side-multi5-cr', side_5_f)

                    side_6_f = slim.convolution2d_transpose(side_6, 16, 64, 32, scope='score-multi6-up')
                    side_6_f = crop_features(side_6_f, im_size)
                    utils.collect_named_outputs(end_points_collection, 'xfcn/side-multi6-cr', side_6_f)

                concat_side = tf.concat([side_2_f, side_3_f, side_4_f, side_5_f, side_6_f], axis=3)
                net = slim.conv2d(concat_side, 1, [1, 1], scope='upscore-fuse')

        end_points = utils.convert_collection_to_dict(end_points_collection)

        return net, end_points


def parameter_lr():
    """Specify the relative learning rate for every parameter. The final learning rate
    in every parameter will be the one defined here multiplied by the global one
    Args:
    Returns:
    Dictionary with the relative learning rate for every parameter
    """

    vars_corresp = dict()

    vars_corresp['xfcn/score-dsn_2/weights'] = 0.1
    vars_corresp['xfcn/score-dsn_2/biases'] = 0.2
    vars_corresp['xfcn/score-dsn_3/weights'] = 0.1
    vars_corresp['xfcn/score-dsn_3/biases'] = 0.2
    vars_corresp['xfcn/score-dsn_4/weights'] = 0.1
    vars_corresp['xfcn/score-dsn_4/biases'] = 0.2
    vars_corresp['xfcn/score-dsn_5/weights'] = 0.1
    vars_corresp['xfcn/score-dsn_5/biases'] = 0.2
    vars_corresp['xfcn/score-dsn_6/weights'] = 0.1
    vars_corresp['xfcn/score-dsn_6/biases'] = 0.2

    vars_corresp['xfcn/upscore-fuse/weights'] = 0.01
    vars_corresp['xfcn/upscore-fuse/biases'] = 0.02

    return vars_corresp