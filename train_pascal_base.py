"""
Reference: https://github.com/scaelles/OSVOS-TensorFlow
"""
import os
import tensorflow as tf
slim = tf.contrib.slim
from utils import models
from utils.load_data_pascal_base import Dataset

# User defined parameters
gpu_id = 0

# Training parameters
imagenet_ckpt = 'weights/imagenet_pretrain_weights/xception_ckpt_new'
logs_path = os.path.join('weights', 'pascal_base_train_weights')
store_memory = True
data_aug = True
iter_mean_grad = 10

max_training_iters = 25000
save_step = 5000

test_image = None
display_step = 10
learning_rate = 1e-6


# Define Dataset
# train_file = 'pretrain_pascal_and_benchmark_new.txt'
# train_file = 'pretrain_benchmark.txt'
train_file = 'datasets/pretrain_benchmark_reduced.txt'
# small dataset txt file for fast debugging
# train_file = 'datasets/test_algorithm_pretrain_benchmark_reduced.txt'

dataset = Dataset(train_file, None, './datasets/pascal_dataset',
                  store_memory=store_memory, data_aug=data_aug)

# Train the network
with tf.Graph().as_default():
    with tf.device('/gpu:' + str(gpu_id)):
        global_step = tf.Variable(0, name='global_step', trainable=False)
        models.pre_train(dataset, imagenet_ckpt, 1, learning_rate, logs_path, max_training_iters, save_step,
                           display_step, global_step, finetune=0, iter_mean_grad=iter_mean_grad, test_image_path=test_image,
                           ckpt_name='pascal_train', dropout_rate=0.5)
