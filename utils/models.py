"""
References: https://github.com/scaelles/OSVOS-TensorFlow
"""
from __future__ import print_function
import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers.python.layers import utils
import sys
from datetime import datetime
import os
import scipy.misc
from PIL import Image
import six

slim = tf.contrib.slim


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


def xfcn_arg_scope(weight_decay=0.0001,
                    batch_norm_decay=0.9997,
                    batch_norm_epsilon=0.001):
    """Defines the xfcn arg scope.
    Args:
    weight_decay: The l2 regularization coefficient.
    Returns:
    An arg_scope.
    """
    with slim.arg_scope([slim.convolution2d_transpose],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.random_normal_initializer(stddev=0.001),
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        biases_initializer=tf.zeros_initializer(),
                        biases_regularizer=None,
                        padding='SAME'):
        with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                            weights_regularizer=slim.l2_regularizer(weight_decay),
                            biases_initializer=None,
                            activation_fn=None):
            with slim.arg_scope([slim.batch_norm],
                                decay=batch_norm_decay,
                                epsilon=batch_norm_epsilon) as arg_sc:
                return arg_sc


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
        with slim.arg_scope([slim.separable_conv2d], depth_multiplier=1), \
             slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                            padding='SAME', activation_fn=None,
                            biases_initializer=None,
                            weights_regularizer=slim.l2_regularizer(0.0001),
                            outputs_collections=end_points_collection), \
             slim.arg_scope([slim.batch_norm], is_training=True, decay=0.9997, epsilon=0.001):
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
            net = slim.separable_conv2d(net, 128, [3, 3],
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
            with slim.arg_scope([slim.conv2d],
                                activation_fn=None,
                                weights_initializer=tf.random_normal_initializer(stddev=0.001),
                                weights_regularizer=slim.l2_regularizer(0.0001),
                                biases_initializer=tf.zeros_initializer(),
                                biases_regularizer=None):
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
                                    activation_fn=None, biases_initializer=None, padding='VALID',
                                    outputs_collections=end_points_collection, trainable=False):
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

        end_points = slim.utils.convert_collection_to_dict(end_points_collection)

        return net, end_points


def upsample_filt(size):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)


# Set deconvolutional layers to compute bilinear interpolation
def interp_surgery(variables):
    interp_tensors = []
    for v in variables:
        if '-up' in v.name:
            h, w, k, m = v.get_shape()
            tmp = np.zeros((m, k, h, w))
            if m != k:
                raise ValueError('input + output channels need to be the same')
            if h != w:
                raise ValueError('filters need to be square')
            up_filter = upsample_filt(int(h))
            tmp[range(m), range(k), :, :] = up_filter
            interp_tensors.append(tf.assign(v, tmp.transpose((2, 3, 1, 0)), validate_shape=True, use_locking=True))
    return interp_tensors


# data augmentation are done in those "load_data*.py" files
def preprocess_img(image):
    """Preprocess the image to adapt it to network requirements
    Args:
    Image we want to input the network (W,H,3) numpy array
    Returns:
    Image ready to input the network (1,W,H,3)
    """
    if type(image) is not np.ndarray:
        image = np.array(Image.open(image), dtype=np.uint8)
    # BGR to RGB
    in_ = image[:, :, ::-1]
    # image centralization
    # They are the mean color values of BSDS500 dataset
    in_ = np.subtract(in_, np.array((104.00699, 116.66877, 122.67892), dtype=np.float32))
    # in_ = tf.subtract(tf.cast(in_, tf.float32), np.array((104.00699, 116.66877, 122.67892), dtype=np.float32))
    # (W,H,3) to (1,W,H,3)
    in_ = np.expand_dims(in_, axis=0)
    # in_ = tf.expand_dims(in_, 0)
    return in_


def preprocess_labels(label):
    """Preprocess the labels to adapt them to the loss computation requirements
    Args:
    Label corresponding to the input image (W,H) numpy array
    Returns:
    Label ready to compute the loss (1,W,H,1)
    """
    if type(label) is not np.ndarray:
        label = np.array(Image.open(label).split()[0], dtype=np.uint8)
    max_mask = np.max(label) * 0.5
    # True False matrix
    label = np.greater(label, max_mask)
    # (W,H) to (B,W,H,1)
    label = np.expand_dims(np.expand_dims(label, axis=0), axis=3)
    # label = tf.cast(np.array(label), tf.float32)
    # max_mask = tf.multiply(tf.reduce_max(label), 0.5)
    # label = tf.cast(tf.greater(label, max_mask), tf.float32)
    # label = tf.expand_dims(tf.expand_dims(label, 0), 3)
    return label


def load_imagenet(ckpt_path):
    """Initialize the network parameters for our xception-lite using ImageNet pretrained weight
    Args:
    Path to the checkpoint
    Returns:
    Function that takes a session and initializes the network
    """
    # ckpt_path: the full path to the model checkpoint (pre-trained model)
    # vars_corresp: A list of `Variable` objects or a dictionary mapping names in the
    # checkpoint (pre-trained model) to the corresponding variables to initialize.
    reader = tf.train.NewCheckpointReader(ckpt_path)
    var_to_shape_map = reader.get_variable_to_shape_map()

    vars_corresp = dict()

    for v in var_to_shape_map:
        if "entry_flow" in v and 'gamma' not in v and 'depthwise/BatchNorm' not in v and 'Momentum' not in v:
            vars_corresp[v] = slim.get_model_variables('xfcn/' + v)[0]
        elif "middle_flow" in v and 'gamma' not in v and 'depthwise/BatchNorm' not in v and 'Momentum' not in v:
            for i in range(1, 3):
                if 'unit_{}/'.format(i) in v:
                    vars_corresp[v] = slim.get_model_variables('xfcn/' + v)[0]
        elif "exit_flow" in v and 'gamma' not in v and 'depthwise/BatchNorm' not in v and 'Momentum' not in v:
            if 'block1/' in v:
                vars_corresp[v] = slim.get_model_variables('xfcn/' + v)[0]
        elif 'shortcut' in v and 'Momentum' not in v and 'gamma' not in v:
            vars_corresp[v] = slim.get_model_variables('xfcn/' + v)[0]

    init_fn = slim.assign_from_checkpoint_fn(
        ckpt_path,
        vars_corresp)
    return init_fn


def class_balanced_cross_entropy_loss(output, label):
    """Define the class balanced cross entropy loss to train the network
    Args:
    output: Output of the network
    label: Ground truth label
    Returns:
    Tensor that evaluates the loss
    """
    labels = tf.cast(tf.greater(label, 0.5), tf.float32)
    num_labels_pos = tf.reduce_sum(labels)
    num_labels_neg = tf.reduce_sum(1.0 - labels)
    num_total = num_labels_pos + num_labels_neg
    output_gt_zero = tf.cast(tf.greater_equal(output, 0), tf.float32)
    loss_val = tf.multiply(output, (labels - output_gt_zero)) - tf.log(
        1 + tf.exp(output - 2 * tf.multiply(output, output_gt_zero)))

    loss_pos = tf.reduce_sum(-tf.multiply(labels, loss_val))
    loss_neg = tf.reduce_sum(-tf.multiply(1.0 - labels, loss_val))

    final_loss = num_labels_neg / num_total * loss_pos + num_labels_pos / num_total * loss_neg

    return final_loss


def parameter_lr():
    """Specify the relative learning rate for every parameter. The final learning rate
    in every parameter will be the one defined here multiplied by the global one
    Args:
    Returns:
    Dictionary with the relative learning rate for every parameter
    """

    vars_corresp = dict()

    # XceptionNet-65
    # Entry flow
    # block 0
    vars_corresp['xfcn/xception_65/entry_flow/conv1_1/weights'] = 1
    vars_corresp['xfcn/xception_65/entry_flow/conv1_1/BatchNorm/beta'] = 1
    vars_corresp['xfcn/xception_65/entry_flow/conv1_2/weights'] = 1
    vars_corresp['xfcn/xception_65/entry_flow/conv1_2/BatchNorm/beta'] = 1
    vars_corresp['xfcn/xception_65/entry_flow/block1/unit_1/xception_module/shortcut/weights'] = 1
    vars_corresp['xfcn/xception_65/entry_flow/block1/unit_1/xception_module/shortcut/BatchNorm/beta'] = 1
    # block 1
    vars_corresp[
        'xfcn/xception_65/entry_flow/block1/unit_1/xception_module/separable_conv1_depthwise/depthwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/entry_flow/block1/unit_1/xception_module/separable_conv1_depthwise/pointwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/entry_flow/block1/unit_1/xception_module/separable_conv1_pointwise/BatchNorm/beta'] = 1
    vars_corresp[
        'xfcn/xception_65/entry_flow/block1/unit_1/xception_module/separable_conv2_depthwise/depthwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/entry_flow/block1/unit_1/xception_module/separable_conv2_depthwise/pointwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/entry_flow/block1/unit_1/xception_module/separable_conv2_pointwise/BatchNorm/beta'] = 1
    vars_corresp[
        'xfcn/xception_65/entry_flow/block1/unit_1/xception_module/separable_conv3_depthwise/depthwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/entry_flow/block1/unit_1/xception_module/separable_conv3_depthwise/pointwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/entry_flow/block1/unit_1/xception_module/separable_conv3_pointwise/BatchNorm/beta'] = 1
    vars_corresp['xfcn/xception_65/entry_flow/block2/unit_1/xception_module/shortcut/weights'] = 1
    vars_corresp['xfcn/xception_65/entry_flow/block2/unit_1/xception_module/shortcut/BatchNorm/beta'] = 1
    # block 2
    vars_corresp[
        'xfcn/xception_65/entry_flow/block2/unit_1/xception_module/separable_conv1_depthwise/depthwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/entry_flow/block2/unit_1/xception_module/separable_conv1_depthwise/pointwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/entry_flow/block2/unit_1/xception_module/separable_conv1_pointwise/BatchNorm/beta'] = 1
    vars_corresp[
        'xfcn/xception_65/entry_flow/block2/unit_1/xception_module/separable_conv2_depthwise/depthwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/entry_flow/block2/unit_1/xception_module/separable_conv2_depthwise/pointwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/entry_flow/block2/unit_1/xception_module/separable_conv2_pointwise/BatchNorm/beta'] = 1
    vars_corresp[
        'xfcn/xception_65/entry_flow/block2/unit_1/xception_module/separable_conv3_depthwise/depthwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/entry_flow/block2/unit_1/xception_module/separable_conv3_depthwise/pointwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/entry_flow/block2/unit_1/xception_module/separable_conv3_pointwise/BatchNorm/beta'] = 1
    vars_corresp['xfcn/xception_65/entry_flow/block3/unit_1/xception_module/shortcut/weights'] = 1
    vars_corresp['xfcn/xception_65/entry_flow/block3/unit_1/xception_module/shortcut/BatchNorm/beta'] = 1
    # block 3
    vars_corresp[
        'xfcn/xception_65/entry_flow/block3/unit_1/xception_module/separable_conv1_depthwise/depthwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/entry_flow/block3/unit_1/xception_module/separable_conv1_depthwise/pointwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/entry_flow/block3/unit_1/xception_module/separable_conv1_pointwise/BatchNorm/beta'] = 1
    vars_corresp[
        'xfcn/xception_65/entry_flow/block3/unit_1/xception_module/separable_conv2_depthwise/depthwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/entry_flow/block3/unit_1/xception_module/separable_conv2_depthwise/pointwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/entry_flow/block3/unit_1/xception_module/separable_conv2_pointwise/BatchNorm/beta'] = 1
    vars_corresp[
        'xfcn/xception_65/entry_flow/block3/unit_1/xception_module/separable_conv3_depthwise/depthwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/entry_flow/block3/unit_1/xception_module/separable_conv3_depthwise/pointwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/entry_flow/block3/unit_1/xception_module/separable_conv3_pointwise/BatchNorm/beta'] = 1

    # middle flow
    # 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_1/xception_module/separable_conv1_depthwise/depthwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_1/xception_module/separable_conv1_depthwise/pointwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_1/xception_module/separable_conv1_pointwise/BatchNorm/beta'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_1/xception_module/separable_conv2_depthwise/depthwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_1/xception_module/separable_conv2_depthwise/pointwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_1/xception_module/separable_conv2_pointwise/BatchNorm/beta'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_1/xception_module/separable_conv3_depthwise/depthwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_1/xception_module/separable_conv3_depthwise/pointwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_1/xception_module/separable_conv3_pointwise/BatchNorm/beta'] = 1
    # 2
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_2/xception_module/separable_conv1_depthwise/depthwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_2/xception_module/separable_conv1_depthwise/pointwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_2/xception_module/separable_conv1_pointwise/BatchNorm/beta'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_2/xception_module/separable_conv2_depthwise/depthwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_2/xception_module/separable_conv2_depthwise/pointwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_2/xception_module/separable_conv2_pointwise/BatchNorm/beta'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_2/xception_module/separable_conv3_depthwise/depthwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_2/xception_module/separable_conv3_depthwise/pointwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_2/xception_module/separable_conv3_pointwise/BatchNorm/beta'] = 1
    # 3
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_3/xception_module/separable_conv1_depthwise/depthwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_3/xception_module/separable_conv1_depthwise/pointwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_3/xception_module/separable_conv1_pointwise/BatchNorm/beta'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_3/xception_module/separable_conv2_depthwise/depthwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_3/xception_module/separable_conv2_depthwise/pointwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_3/xception_module/separable_conv2_pointwise/BatchNorm/beta'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_3/xception_module/separable_conv3_depthwise/depthwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_3/xception_module/separable_conv3_depthwise/pointwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_3/xception_module/separable_conv3_pointwise/BatchNorm/beta'] = 1
    # 4
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_4/xception_module/separable_conv1_depthwise/depthwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_4/xception_module/separable_conv1_depthwise/pointwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_4/xception_module/separable_conv1_pointwise/BatchNorm/beta'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_4/xception_module/separable_conv2_depthwise/depthwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_4/xception_module/separable_conv2_depthwise/pointwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_4/xception_module/separable_conv2_pointwise/BatchNorm/beta'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_4/xception_module/separable_conv3_depthwise/depthwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_4/xception_module/separable_conv3_depthwise/pointwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_4/xception_module/separable_conv3_pointwise/BatchNorm/beta'] = 1
    # 5
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_5/xception_module/separable_conv1_depthwise/depthwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_5/xception_module/separable_conv1_depthwise/pointwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_5/xception_module/separable_conv1_pointwise/BatchNorm/beta'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_5/xception_module/separable_conv2_depthwise/depthwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_5/xception_module/separable_conv2_depthwise/pointwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_5/xception_module/separable_conv2_pointwise/BatchNorm/beta'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_5/xception_module/separable_conv3_depthwise/depthwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_5/xception_module/separable_conv3_depthwise/pointwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_5/xception_module/separable_conv3_pointwise/BatchNorm/beta'] = 1
    # 6
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_6/xception_module/separable_conv1_depthwise/depthwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_6/xception_module/separable_conv1_depthwise/pointwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_6/xception_module/separable_conv1_pointwise/BatchNorm/beta'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_6/xception_module/separable_conv2_depthwise/depthwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_6/xception_module/separable_conv2_depthwise/pointwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_6/xception_module/separable_conv2_pointwise/BatchNorm/beta'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_6/xception_module/separable_conv3_depthwise/depthwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_6/xception_module/separable_conv3_depthwise/pointwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_6/xception_module/separable_conv3_pointwise/BatchNorm/beta'] = 1
    # 7
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_7/xception_module/separable_conv1_depthwise/depthwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_7/xception_module/separable_conv1_depthwise/pointwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_7/xception_module/separable_conv1_pointwise/BatchNorm/beta'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_7/xception_module/separable_conv2_depthwise/depthwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_7/xception_module/separable_conv2_depthwise/pointwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_7/xception_module/separable_conv2_pointwise/BatchNorm/beta'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_7/xception_module/separable_conv3_depthwise/depthwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_7/xception_module/separable_conv3_depthwise/pointwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_7/xception_module/separable_conv3_pointwise/BatchNorm/beta'] = 1
    # 8
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_8/xception_module/separable_conv1_depthwise/depthwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_8/xception_module/separable_conv1_depthwise/pointwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_8/xception_module/separable_conv1_pointwise/BatchNorm/beta'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_8/xception_module/separable_conv2_depthwise/depthwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_8/xception_module/separable_conv2_depthwise/pointwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_8/xception_module/separable_conv2_pointwise/BatchNorm/beta'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_8/xception_module/separable_conv3_depthwise/depthwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_8/xception_module/separable_conv3_depthwise/pointwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_8/xception_module/separable_conv3_pointwise/BatchNorm/beta'] = 1
    # 9
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_9/xception_module/separable_conv1_depthwise/depthwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_9/xception_module/separable_conv1_depthwise/pointwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_9/xception_module/separable_conv1_pointwise/BatchNorm/beta'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_9/xception_module/separable_conv2_depthwise/depthwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_9/xception_module/separable_conv2_depthwise/pointwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_9/xception_module/separable_conv2_pointwise/BatchNorm/beta'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_9/xception_module/separable_conv3_depthwise/depthwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_9/xception_module/separable_conv3_depthwise/pointwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_9/xception_module/separable_conv3_pointwise/BatchNorm/beta'] = 1
    # 10
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_10/xception_module/separable_conv1_depthwise/depthwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_10/xception_module/separable_conv1_depthwise/pointwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_10/xception_module/separable_conv1_pointwise/BatchNorm/beta'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_10/xception_module/separable_conv2_depthwise/depthwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_10/xception_module/separable_conv2_depthwise/pointwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_10/xception_module/separable_conv2_pointwise/BatchNorm/beta'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_10/xception_module/separable_conv3_depthwise/depthwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_10/xception_module/separable_conv3_depthwise/pointwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_10/xception_module/separable_conv3_pointwise/BatchNorm/beta'] = 1
    # 11
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_11/xception_module/separable_conv1_depthwise/depthwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_11/xception_module/separable_conv1_depthwise/pointwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_11/xception_module/separable_conv1_pointwise/BatchNorm/beta'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_11/xception_module/separable_conv2_depthwise/depthwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_11/xception_module/separable_conv2_depthwise/pointwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_11/xception_module/separable_conv2_pointwise/BatchNorm/beta'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_11/xception_module/separable_conv3_depthwise/depthwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_11/xception_module/separable_conv3_depthwise/pointwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_11/xception_module/separable_conv3_pointwise/BatchNorm/beta'] = 1
    # 12
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_12/xception_module/separable_conv1_depthwise/depthwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_12/xception_module/separable_conv1_depthwise/pointwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_12/xception_module/separable_conv1_pointwise/BatchNorm/beta'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_12/xception_module/separable_conv2_depthwise/depthwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_12/xception_module/separable_conv2_depthwise/pointwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_12/xception_module/separable_conv2_pointwise/BatchNorm/beta'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_12/xception_module/separable_conv3_depthwise/depthwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_12/xception_module/separable_conv3_depthwise/pointwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_12/xception_module/separable_conv3_pointwise/BatchNorm/beta'] = 1
    # 13
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_13/xception_module/separable_conv1_depthwise/depthwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_13/xception_module/separable_conv1_depthwise/pointwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_13/xception_module/separable_conv1_pointwise/BatchNorm/beta'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_13/xception_module/separable_conv2_depthwise/depthwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_13/xception_module/separable_conv2_depthwise/pointwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_13/xception_module/separable_conv2_pointwise/BatchNorm/beta'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_13/xception_module/separable_conv3_depthwise/depthwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_13/xception_module/separable_conv3_depthwise/pointwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_13/xception_module/separable_conv3_pointwise/BatchNorm/beta'] = 1
    # 14
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_14/xception_module/separable_conv1_depthwise/depthwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_14/xception_module/separable_conv1_depthwise/pointwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_14/xception_module/separable_conv1_pointwise/BatchNorm/beta'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_14/xception_module/separable_conv2_depthwise/depthwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_14/xception_module/separable_conv2_depthwise/pointwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_14/xception_module/separable_conv2_pointwise/BatchNorm/beta'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_14/xception_module/separable_conv3_depthwise/depthwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_14/xception_module/separable_conv3_depthwise/pointwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_14/xception_module/separable_conv3_pointwise/BatchNorm/beta'] = 1
    # 15
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_15/xception_module/separable_conv1_depthwise/depthwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_15/xception_module/separable_conv1_depthwise/pointwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_15/xception_module/separable_conv1_pointwise/BatchNorm/beta'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_15/xception_module/separable_conv2_depthwise/depthwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_15/xception_module/separable_conv2_depthwise/pointwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_15/xception_module/separable_conv2_pointwise/BatchNorm/beta'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_15/xception_module/separable_conv3_depthwise/depthwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_15/xception_module/separable_conv3_depthwise/pointwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_15/xception_module/separable_conv3_pointwise/BatchNorm/beta'] = 1
    # 16
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_16/xception_module/separable_conv1_depthwise/depthwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_16/xception_module/separable_conv1_depthwise/pointwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_16/xception_module/separable_conv1_pointwise/BatchNorm/beta'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_16/xception_module/separable_conv2_depthwise/depthwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_16/xception_module/separable_conv2_depthwise/pointwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_16/xception_module/separable_conv2_pointwise/BatchNorm/beta'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_16/xception_module/separable_conv3_depthwise/depthwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_16/xception_module/separable_conv3_depthwise/pointwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/middle_flow/block1/unit_16/xception_module/separable_conv3_pointwise/BatchNorm/beta'] = 1

    # Exit flow
    vars_corresp['xfcn/xception_65/exit_flow/block1/unit_1/xception_module/shortcut/weights'] = 1
    vars_corresp['xfcn/xception_65/exit_flow/block1/unit_1/xception_module/shortcut/BatchNorm/beta'] = 1
    # block 1
    vars_corresp[
        'xfcn/xception_65/exit_flow/block1/unit_1/xception_module/separable_conv1_depthwise/depthwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/exit_flow/block1/unit_1/xception_module/separable_conv1_depthwise/pointwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/exit_flow/block1/unit_1/xception_module/separable_conv1_pointwise/BatchNorm/beta'] = 1
    vars_corresp[
        'xfcn/xception_65/exit_flow/block1/unit_1/xception_module/separable_conv2_depthwise/depthwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/exit_flow/block1/unit_1/xception_module/separable_conv2_depthwise/pointwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/exit_flow/block1/unit_1/xception_module/separable_conv2_pointwise/BatchNorm/beta'] = 1
    vars_corresp[
        'xfcn/xception_65/exit_flow/block1/unit_1/xception_module/separable_conv3_depthwise/depthwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/exit_flow/block1/unit_1/xception_module/separable_conv3_depthwise/pointwise_weights'] = 1
    vars_corresp[
        'xfcn/xception_65/exit_flow/block1/unit_1/xception_module/separable_conv3_pointwise/BatchNorm/beta'] = 1
    # block 2
    # vars_corresp[
    #     'xfcn/xception_65/exit_flow/block2/unit_1/xception_module/separable_conv1_depthwise/depthwise_weights'] = 1
    # vars_corresp[
    #     'xfcn/xception_65/exit_flow/block2/unit_1/xception_module/separable_conv1_depthwise/pointwise_weights'] = 1
    # vars_corresp[
    #     'xfcn/xception_65/exit_flow/block2/unit_1/xception_module/separable_conv1_pointwise/BatchNorm/beta'] = 1
    # vars_corresp[
    #     'xfcn/xception_65/exit_flow/block2/unit_1/xception_module/separable_conv2_depthwise/depthwise_weights'] = 1
    # vars_corresp[
    #     'xfcn/xception_65/exit_flow/block2/unit_1/xception_module/separable_conv2_depthwise/pointwise_weights'] = 1
    # vars_corresp[
    #     'xfcn/xception_65/exit_flow/block2/unit_1/xception_module/separable_conv2_pointwise/BatchNorm/beta'] = 1
    # vars_corresp[
    #     'xfcn/xception_65/exit_flow/block2/unit_1/xception_module/separable_conv3_depthwise/depthwise_weights'] = 1
    # vars_corresp[
    #     'xfcn/xception_65/exit_flow/block2/unit_1/xception_module/separable_conv3_depthwise/pointwise_weights'] = 1
    # vars_corresp[
    #     'xfcn/xception_65/exit_flow/block2/unit_1/xception_module/separable_conv3_pointwise/BatchNorm/beta'] = 1
    # xceptionnet65 结束

    vars_corresp['xfcn/conv2_2_16/weights'] = 1
    vars_corresp['xfcn/conv2_2_16/biases'] = 2
    vars_corresp['xfcn/conv3_3_16/weights'] = 1
    vars_corresp['xfcn/conv3_3_16/biases'] = 2
    vars_corresp['xfcn/conv4_3_16/weights'] = 1
    vars_corresp['xfcn/conv4_3_16/biases'] = 2

    vars_corresp['xfcn/conv6_3_16/weights'] = 1
    vars_corresp['xfcn/conv6_3_16/biases'] = 2
    vars_corresp['xfcn/conv7_3_16/weights'] = 1
    vars_corresp['xfcn/conv7_3_16/biases'] = 2
    vars_corresp['xfcn/conv8_3_16/weights'] = 1
    vars_corresp['xfcn/conv8_3_16/biases'] = 2
    vars_corresp['xfcn/conv9_3_16/weights'] = 1
    vars_corresp['xfcn/conv9_3_16/biases'] = 2
    vars_corresp['xfcn/conv10_3_16/weights'] = 1
    vars_corresp['xfcn/conv10_3_16/biases'] = 2

    vars_corresp['xfcn/conv5_3_16/weights'] = 1
    vars_corresp['xfcn/conv5_3_16/biases'] = 2

    vars_corresp['xfcn/score-dsn_2/weights'] = 0.1
    vars_corresp['xfcn/score-dsn_2/biases'] = 0.2
    vars_corresp['xfcn/score-dsn_3/weights'] = 0.1
    vars_corresp['xfcn/score-dsn_3/biases'] = 0.2
    vars_corresp['xfcn/score-dsn_4/weights'] = 0.1
    vars_corresp['xfcn/score-dsn_4/biases'] = 0.2

    vars_corresp['xfcn/score-dsn_6/weights'] = 0.1
    vars_corresp['xfcn/score-dsn_6/biases'] = 0.2
    vars_corresp['xfcn/score-dsn_7/weights'] = 0.1
    vars_corresp['xfcn/score-dsn_7/biases'] = 0.2
    vars_corresp['xfcn/score-dsn_8/weights'] = 0.1
    vars_corresp['xfcn/score-dsn_8/biases'] = 0.2
    vars_corresp['xfcn/score-dsn_9/weights'] = 0.1
    vars_corresp['xfcn/score-dsn_9/biases'] = 0.2
    vars_corresp['xfcn/score-dsn_10/weights'] = 0.1
    vars_corresp['xfcn/score-dsn_10/biases'] = 0.2

    vars_corresp['xfcn/score-dsn_5/weights'] = 0.1
    vars_corresp['xfcn/score-dsn_5/biases'] = 0.2

    vars_corresp['xfcn/upscore-fuse/weights'] = 0.01
    vars_corresp['xfcn/upscore-fuse/biases'] = 0.02
    # vars_corresp['xfcn/upscore-fuse_1/weights'] = 0.01
    # vars_corresp['xfcn/upscore-fuse_1/biases'] = 0.02
    # vars_corresp['xfcn/upscore-fuse_2/weights'] = 0.01
    # vars_corresp['xfcn/upscore-fuse_2/biases'] = 0.02
    return vars_corresp


def _train(dataset, initial_ckpt, supervison, learning_rate, logs_path, max_training_iters, save_step, display_step,
           global_step, iter_mean_grad=1, batch_size=1, momentum=0.9, config=None, finetune=1,
           test_image_path=None, ckpt_name="xfcn", dropout_rate=1.0):
    """Train xfcn
    Args:
    dataset: Reference to a Dataset object instance
    initial_ckpt: Path to the checkpoint (different pre-trained weight) to initialize the network
    supervison: Level of the side outputs supervision: 1-Strong 2-Weak 3-No supervision
    learning_rate: Value for the learning rate.
    logs_path: Path to store the checkpoints (different weights)
    max_training_iters: Number of training iterations
    save_step: A checkpoint (weight) will be created every save_steps
    display_step: Information of the training will be displayed every display_steps
    global_step: Reference to a Variable that keeps track of the training steps
    iter_mean_grad: Number of gradient computations that are average before updating the weights
    batch_size: Size of the training batch
    momentum: Value of the momentum parameter for the Momentum optimizer
    config: Reference to a Configuration object used in the creation of a Session
    finetune: decide which pre-trained model should be selected
    test_image_path: If image path provided, every save_step the result of the network with this image is stored
    dropout_rate: dropout rate in XFCN
    Returns:
    """

    model_name = os.path.join(logs_path, ckpt_name + ".ckpt")
    if config is None:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # config.log_device_placement = True
        config.allow_soft_placement = True

    tf.logging.set_verbosity(tf.logging.INFO)

    # Prepare the input data
    input_image = tf.placeholder(tf.float32, [batch_size, None, None, 3])
    input_label = tf.placeholder(tf.float32, [batch_size, None, None, 1])

    # Create the network
    with slim.arg_scope(xfcn_arg_scope()):
        net, end_points = xfcn(input_image, dropout_rate)

    # Initialize weights from pre-trained (ImageNet) model
    init_weights = None
    if finetune == 0:
        init_weights = load_imagenet(initial_ckpt)

    # Define loss
    with tf.name_scope('losses'):
        if supervison == 1 or supervison == 2:
            dsn_2_loss = class_balanced_cross_entropy_loss(end_points['xfcn/score-dsn_2-cr'], input_label)
            tf.summary.scalar('dsn_2_loss', dsn_2_loss)
            dsn_3_loss = class_balanced_cross_entropy_loss(end_points['xfcn/score-dsn_3-cr'], input_label)
            tf.summary.scalar('dsn_3_loss', dsn_3_loss)
            dsn_4_loss = class_balanced_cross_entropy_loss(end_points['xfcn/score-dsn_4-cr'], input_label)
            tf.summary.scalar('dsn_4_loss', dsn_4_loss)
            dsn_5_loss = class_balanced_cross_entropy_loss(end_points['xfcn/score-dsn_5-cr'], input_label)
            tf.summary.scalar('dsn_5_loss', dsn_5_loss)
            dsn_6_loss = class_balanced_cross_entropy_loss(end_points['xfcn/score-dsn_6-cr'], input_label)
            tf.summary.scalar('dsn_6_loss', dsn_6_loss)

        main_loss = class_balanced_cross_entropy_loss(net, input_label)
        tf.summary.scalar('main_loss', main_loss)

        if supervison == 1:
            output_loss = dsn_2_loss + dsn_3_loss + dsn_4_loss + dsn_5_loss + dsn_6_loss + main_loss
        elif supervison == 2:
            output_loss = 0.5 * dsn_2_loss + 0.5 * dsn_3_loss + 0.5 * dsn_4_loss + 0.5 * dsn_5_loss + 0.5*dsn_6_loss + main_loss
        elif supervison == 3:
            output_loss = main_loss
        else:
            sys.exit('Incorrect supervision id, select 1 for supervision of the side outputs, 2 for weak supervision '
                     'of the side outputs and 3 for no supervision of the side outputs')
        total_loss = output_loss + tf.add_n(tf.losses.get_regularization_losses())
        tf.summary.scalar('total_loss', total_loss)

    # Define optimization method
    with tf.name_scope('optimization'):
        tf.summary.scalar('learning_rate', learning_rate)
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
        grads_and_vars = optimizer.compute_gradients(total_loss)
        with tf.name_scope('grad_accumulator'):
            grad_accumulator = {}
            for ind in range(0, len(grads_and_vars)):
                if grads_and_vars[ind][0] is not None:
                    grad_accumulator[ind] = tf.ConditionalAccumulator(grads_and_vars[ind][0].dtype)
        with tf.name_scope('apply_gradient'):
            layer_lr = parameter_lr()
            grad_accumulator_ops = []
            for var_ind, grad_acc in six.iteritems(grad_accumulator):
                var_name = str(grads_and_vars[var_ind][1].name).split(':')[0]
                var_grad = grads_and_vars[var_ind][0]
                grad_accumulator_ops.append(grad_acc.apply_grad(var_grad * layer_lr[var_name],
                                                                local_step=global_step))
        with tf.name_scope('take_gradients'):
            mean_grads_and_vars = []
            for var_ind, grad_acc in six.iteritems(grad_accumulator):
                mean_grads_and_vars.append(
                    (grad_acc.take_grad(iter_mean_grad), grads_and_vars[var_ind][1]))
            apply_gradient_op = optimizer.apply_gradients(mean_grads_and_vars, global_step=global_step)
    # Log training info
    merged_summary_op = tf.summary.merge_all()

    # Log evolution of test image
    if test_image_path is not None:
        probabilities = tf.nn.sigmoid(net)
        img_summary = tf.summary.image("Output probabilities", probabilities, max_outputs=1)
    # Initialize variables
    init = tf.global_variables_initializer()

    # Run Session
    with tf.Session(config=config) as sess:
        print('Init variable')
        sess.run(init)

        # op to write logs to Tensorboard
        summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

        # Create saver to manage checkpoints
        saver = tf.train.Saver(max_to_keep=None)

        # Load pre-trained model
        if finetune == 0:
            print('Initializing from xceptnet imagenet model...')
            init_weights(sess)
        else:
            print('Initializing from pre-trained model...')
            var_list = []
            for var in tf.global_variables():
                var_type = var.name
                # not import global_step variable
                if 'global_step' not in var_type:
                    var_list.append(var)
            saver_res = tf.train.Saver(var_list=var_list)
            saver_res.restore(sess, initial_ckpt)
        step = global_step.eval() + 1
        sess.run(interp_surgery(tf.global_variables()))
        print('Weights initialized')

        print('Start training')
        while step < max_training_iters + 1:
            # Average the gradient
            for _ in range(0, iter_mean_grad):
                batch_image, batch_label = dataset.next_batch(batch_size, 'train')
                image = preprocess_img(batch_image[0])
                label = preprocess_labels(batch_label[0])
                run_res = sess.run([total_loss, merged_summary_op] + grad_accumulator_ops,
                                   feed_dict={input_image: image, input_label: label})
                batch_loss = run_res[0]
                summary = run_res[1]

            # Apply the gradients
            sess.run(apply_gradient_op)  # Momentum updates here its statistics

            # Save summary reports
            summary_writer.add_summary(summary, step)

            # Display training status
            if step % display_step == 0:
                print("{} Iter {}: Training Loss = {:.4f}".format(datetime.now(), step, batch_loss), file=sys.stderr)

            # Save a checkpoint
            if step % save_step == 0:
                if test_image_path is not None:
                    curr_output = sess.run(img_summary, feed_dict={input_image: preprocess_img(test_image_path)})
                    summary_writer.add_summary(curr_output, step)
                save_path = saver.save(sess, model_name, global_step=global_step)
                print("Model saved in file: %s" % save_path)

            step += 1

        if (step - 1) % save_step != 0:
            save_path = saver.save(sess, model_name, global_step=global_step)
            print("Model saved in file: %s" % save_path)

        print('Finished training.')


def pre_train(dataset, initial_ckpt, supervison, learning_rate, logs_path, max_training_iters, save_step,
                 display_step, global_step, finetune, iter_mean_grad=1, momentum=0.9,
                 config=None, test_image_path=None, ckpt_name="xfcn", dropout_rate=1.0):
    """Train xfcn parent network
    Args:
    See _train()
    Returns:
    """
    batch_size = 1
    _train(dataset, initial_ckpt, supervison, learning_rate, logs_path, max_training_iters, save_step, display_step,
           global_step, iter_mean_grad, batch_size, momentum, config, finetune, test_image_path,
           ckpt_name, dropout_rate=dropout_rate)


def train_finetune(dataset, initial_ckpt, supervison, learning_rate, logs_path, max_training_iters, save_step,
                   display_step, global_step, finetune, iter_mean_grad=1, batch_size=1, momentum=0.9,
                   config=None, test_image_path=None, ckpt_name="xfcn", dropout_rate=1.0):
    """Finetune xfcn
    Args:
    See _train()
    Returns:
    """
    _train(dataset, initial_ckpt, supervison, learning_rate, logs_path, max_training_iters, save_step, display_step,
           global_step, iter_mean_grad, batch_size, momentum, config, finetune, test_image_path,
           ckpt_name, dropout_rate=dropout_rate)


def test(dataset, checkpoint_file, result_path, config=None, dropout_rate=1.0):
    """Test one sequence
    Args:
    dataset: Reference to a Dataset object instance
    checkpoint_path: Path of the checkpoint to use for the evaluation
    result_path: Path to save the output images
    config: Reference to a Configuration object used in the creation of a Session
    dropout_rate: keep dropout_rate is 1.0 in testing
    Returns:
    """
    if config is None:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
    tf.logging.set_verbosity(tf.logging.INFO)

    # Input data
    batch_size = 1
    input_image = tf.placeholder(tf.float32, [batch_size, None, None, 3])

    # Create the cnn
    with slim.arg_scope(xfcn_arg_scope()):
        net, end_points = xfcn(input_image, dropout_rate)
    probabilities = tf.nn.sigmoid(net)

    # Create a saver to load the network
    saver = tf.train.Saver([v for v in tf.global_variables() if '-up' not in v.name and '-cr' not in v.name])

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(interp_surgery(tf.global_variables()))
        saver.restore(sess, checkpoint_file)

        if not os.path.exists(result_path):
            os.makedirs(result_path)

        # test data loading and augmentation
        res_list = []
        idx = 0
        old_idx = 0
        for frame in range(0, dataset.get_test_size()):
            # images, paths
            img, curr_img, aug_num = dataset.next_batch(batch_size, 'test')
            curr_frame_orig_name = os.path.split(curr_img[0])[1]
            curr_frame = os.path.splitext(curr_frame_orig_name)[0] + '.png'
            image = preprocess_img(img[0])
            res = sess.run(probabilities, feed_dict={input_image: image})
            res = res.astype(np.float32)[0, :, :, 0]
            if idx % aug_num == 1:
                res = np.fliplr(res)
            res_list.append(res)
            if idx % aug_num == aug_num-1:
                res_average = np.sum(res_list[old_idx:idx+1], axis=0) / float(aug_num)
                old_idx += aug_num
                # optimal threshold in our case
                res_np = res_average > 162.0 / 255.0
                # # remove outliers
                # res_np = remove_small_objects(res_np, min_size=2000, connectivity=2)
                scipy.misc.imsave(os.path.join(result_path, curr_frame), res_np.astype(np.float32))
                print('Saving ' + os.path.join(result_path, curr_frame))
            idx += 1
