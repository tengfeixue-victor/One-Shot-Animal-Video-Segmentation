"""
Reference: https://github.com/scaelles/OSVOS-TensorFlow
"""
import os
import tensorflow as tf
import random
import numpy as np

from utils import models
from utils.load_data_pascal_base import Dataset
from utils.logger import create_logger

# seed
seed = random.randint(1, 100000)

# seed = 0
tf.random.set_seed(seed)
random.seed(seed)
np.random.seed(seed)

# User defined parameters
gpu_id = 0

# Training parameters
imagenet_ckpt = 'weights/imagenet_pretrain_weights/xception_ckpt_new'
logs_path = os.path.join('weights', 'pascal_base_train_weights')
if not os.path.exists(logs_path):
    os.mkdir(logs_path)

store_memory = True
data_aug = True
pretrained_model = True
supervision = 1

iter_mean_grad = 10
max_training_iters = 25000
save_step = 5000

test_image = None
display_step = 10
# learning rate setting
ini_learning_rate = 1e-6
end_learning_rate = 2.5 * 1e-7
batch_size = 1

# log some important info
logger = create_logger(logs_path)
logger.info('The random seed is {}'.format(seed))
logger.info('The max training iteration is {}'.format(max_training_iters))
logger.info('The supervision mode is {}'.format(supervision))
logger.info('Data augmentation is {}'.format(data_aug))

# Define Dataset
# use this one for training
train_file = 'datasets/pretrain_benchmark_reduced.txt'

# # small dataset txt file for fast debugging
# train_file = 'datasets/test_algorithm_pretrain_benchmark_reduced.txt'

dataset = Dataset(train_file, None, './datasets/pascal_extension_dataset',
                  store_memory=store_memory, data_aug=data_aug)

# Train the network
with tf.Graph().as_default():
    with tf.device('/gpu:' + str(gpu_id)):
        global_step = tf.Variable(0, name='global_step', trainable=False)
        learning_rate = tf.compat.v1.train.polynomial_decay(learning_rate=ini_learning_rate, global_step=global_step,
                                                  decay_steps=max_training_iters,
                                                  end_learning_rate=end_learning_rate, power=2)
        models.pre_train(dataset, imagenet_ckpt, supervision, learning_rate, logs_path, max_training_iters, save_step,
                           display_step, global_step, logger, finetune=0, iter_mean_grad=iter_mean_grad, test_image_path=test_image,
                           ckpt_name='pascal_train', dropout_rate=0.5, batch_size=batch_size, pretrained_model=pretrained_model)
