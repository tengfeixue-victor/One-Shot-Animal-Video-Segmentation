"""
https://github.com/scaelles/OSVOS-TensorFlow
"""
import os
import tensorflow as tf
from utils import models
from utils.load_data_davis_objectness import Dataset
import random
import numpy as np
from utils.logger import create_logger
import sys

# seed
seed = random.randint(1, 100000)
# seed = 0
tf.random.set_seed(seed)
random.seed(seed)
np.random.seed(seed)

# User defined parameters
gpu_id = 0

# Training parameters
pascal_base_iterations = 25000

pascal_base_ckpt = 'weights/pascal_base_train_weights/pascal_train.ckpt-{}'.format(pascal_base_iterations)

logs_path = os.path.join('weights', 'objectness_weights')
if not os.path.exists(logs_path):
    os.mkdir(logs_path)

store_memory = True
data_aug = True
test_image = None
supervision = 1

iter_mean_grad = 10
max_training_iters = 45000
save_step = 5000

display_step = 10
# learning rate setting
ini_learning_rate = 1e-6
end_learning_rate = 2.5 * 1e-7
batch_size = 1

# log some important info
logger = create_logger(logs_path)
logger.info('The random seed is {}'.format(seed))
logger.info('The pascal base training iteration is {}'.format(pascal_base_iterations))
logger.info('The max objectness training iteration is {}'.format(max_training_iters))
logger.info('The supervision mode is {}'.format(supervision))
logger.info('Data augmentation is {}'.format(data_aug))

# Define Dataset
train_file = 'datasets/davis2016_trainset.txt'
# # small dataset txt file for fast debugging
# train_file = 'datasets/test_algorithm_davis2016_trainset.txt'

dataset = Dataset(train_file, None, './datasets/DAVIS2016_train_dataset', store_memory=store_memory, data_aug=data_aug)

# Train the network
with tf.Graph().as_default():
    with tf.device('/gpu:' + str(gpu_id)):
        global_step = tf.Variable(pascal_base_iterations, name='global_step', trainable=False)
        learning_rate = tf.compat.v1.train.polynomial_decay(learning_rate=ini_learning_rate, global_step=global_step,
                                                  decay_steps=max_training_iters,
                                                  end_learning_rate=end_learning_rate, power=2)
        models.pre_train(dataset, pascal_base_ckpt, supervision, learning_rate, logs_path, max_training_iters, save_step,
                            display_step, global_step, logger, finetune=1, iter_mean_grad=iter_mean_grad,
                            test_image_path=test_image, ckpt_name='objectness_weights', dropout_rate=0.5,
                            batch_size=batch_size)
