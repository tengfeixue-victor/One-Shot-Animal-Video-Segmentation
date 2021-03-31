"""
References: https://github.com/scaelles/OSVOS-TensorFlow
"""
from __future__ import print_function
import os
import random
import tensorflow as tf
slim = tf.contrib.slim
from utils import models
from utils.load_data_finetune import Dataset

# User defined path parameters
# finetuning (one label) and testing dataset
sequence_images_path = './datasets/finetune_test_dataset/JPEGImages/480p'
sequence_names = os.listdir(sequence_images_path)

# Get the best frame selection from BubblNet
bub_frame_path = './datasets/bubbleNet_data/rawData'


def create_non_exist_file(non_exist_file):
    """Create the file when it does not exist"""
    if not os.path.exists(non_exist_file):
        os.mkdir(non_exist_file)


def select_optimal_frame(seq_name):
    """Use the optimal frame from BubbleNet selection for fine-tuning"""
    # # Select from BN0 or BNLF
    # frame_txt = os.path.join(bub_frame_path, seq_name, 'frame_selection/all.txt')
    # # Select from BN0
    # frame_txt = os.path.join(bub_frame_path, seq_name, 'frame_selection/BN0.txt')

    # Select from BNLF (BNLF has better results on DAVIS animal than BN0)
    frame_txt = os.path.join(bub_frame_path, seq_name, 'frame_selection/BNLF.txt')
    frame_file = open(frame_txt, 'r')
    frame_nums = frame_file.readlines()
    # The following code is used to extract the name of frame selection
    # refer to the txt file in './datasets/bubbleNet_data/rawData/frame_selection' for your information
    if len(frame_nums) == 3:
        frame_random_jpg = frame_nums[2][:9]
        frame_random_png = frame_nums[2][:5] + '.png'
    # when two bubblenet models select the different frames, the txt file will have 5 lines
    elif len(frame_nums) == 5:
        frame_suggestion1_jpg = frame_nums[2][:9]
        frame_suggestion1_png = frame_nums[2][:5] + '.png'
        frame_suggestion2_jpg = frame_nums[4][:9]
        frame_suggestion2_png = frame_nums[4][:5] + '.png'
        frame_random_lst = random.choice(
            [[frame_suggestion1_jpg, frame_suggestion1_png], [frame_suggestion2_jpg, frame_suggestion2_png]])
        # 这里每行都会包含一个换行符号，所以只取前9个字符，即'00000.jpg'
        frame_random_jpg = frame_random_lst[0][:9]
        frame_random_png = frame_random_lst[1][:9]
    else:
        raise ValueError("frame file from BubbleNet is not correct")

    return frame_random_jpg, frame_random_png


def train_test(video_path_names):
    for sequence_name in video_path_names:
        seq_name = "{}".format(sequence_name)
        gpu_id = 0

        # Train and test parameters
        # training and testing or testing only
        train_model = True
        objectness_steps = 45000
        # The path to obtain weights from objectness training
        objectness_path = os.path.join('weights', 'objectness_weights', 'objectness_weights.ckpt-{}'.format(objectness_steps))
        # The path to save weights of fine tuning
        logs_path_base = os.path.join('weights', 'fine_tune_weights')
        create_non_exist_file(logs_path_base)
        logs_path = os.path.join(logs_path_base, seq_name)
        # the epochs for fine tuning
        max_training_iters = 200
        # max_training_iters = 10
        # test data augmentation
        test_aug = True

        # Define Dataset
        # the video for tesing
        test_frames = sorted(
            os.listdir(os.path.join('datasets', 'finetune_test_dataset', 'JPEGImages', '480p', seq_name)))
        test_imgs = [os.path.join('datasets', 'finetune_test_dataset', 'JPEGImages', '480p', seq_name, frame) for frame
                     in test_frames]

        # result paths
        create_non_exist_file('results')
        result_path_base = os.path.join('results', 'segmentation')
        create_non_exist_file(result_path_base)
        result_path = os.path.join(result_path_base, seq_name)
        create_non_exist_file(result_path)
        # fmap_path = os.path.join('results', 'segmentation', 'feature_map', seq_name)
        # red_mask_path = os.path.join('results', 'red_mask', seq_name)
        # create_non_exist_file(fmap_path)
        # create_non_exist_file(red_mask_path)

        if train_model:
            # # Train on the first frame
            # train_imgs = [os.path.join('datasets', 'finetune_test_dataset',
            # 'JPEGImages', '480p', seq_name, '00000.jpg') + ' ' + os.path.join('datasets', 'finetune_test_dataset',
            # 'Annotations', '480p', seq_name, '00000.png')]

            # BubbleNet selection: one optimal frame
            frame_random_jpg, frame_random_png = select_optimal_frame(seq_name)
            selected_image = os.path.join('datasets', 'finetune_test_dataset', 'JPEGImages', '480p', seq_name,
                                          frame_random_jpg)
            selected_mask = os.path.join('datasets', 'finetune_test_dataset', 'Annotations', '480p', seq_name,
                                         frame_random_png)
            train_imgs = [selected_image + ' ' + selected_mask]
            print('select frame {} in folder {}'.format(frame_random_jpg, seq_name))

            # test augmentation is on
            dataset = Dataset(train_imgs, test_imgs, './', data_aug=True, test_aug=test_aug)

        # testing only
        else:
            # test augmentation is on
            dataset = Dataset(None, test_imgs, './', test_aug=test_aug)

        # Train the network
        if train_model:
            # More training parameters
            learning_rate = 1e-7
            save_step = max_training_iters
            # no side supervision
            side_supervision = 3
            display_step = 10
            with tf.Graph().as_default():
                with tf.device('/gpu:' + str(gpu_id)):
                    # global_step is related to the name of cpkt file
                    global_step = tf.Variable(0, name='global_step', trainable=False)
                    models.train_finetune(dataset, objectness_path, side_supervision, learning_rate, logs_path,
                                          max_training_iters, save_step, display_step, global_step, finetune=1,
                                          iter_mean_grad=1, ckpt_name=seq_name, dropout_rate=1.0)

        # Test the network
        with tf.Graph().as_default():
            with tf.device('/gpu:' + str(gpu_id)):
                # No fine-tuning
                checkpoint_path = os.path.join('weights/fine_tune_weights/', seq_name,
                                               seq_name + '.ckpt-' + str(max_training_iters))
                # generate results images(binary) to the results path
                models.test(dataset, checkpoint_path, result_path)
                # models.test(dataset, checkpoint_path, result_path, feature_map_path=fmap_path)


if __name__ == '__main__':
    train_test(sequence_names)
