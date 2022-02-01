# One-Shot-Animal-Video-Segmentation

This repository makes available the source code, datasets, pre-trained weights for the work, "One-shot Learning-based Animal Video Segmentation", which is accepted by IEEE Transactions on Industrial Informatics.

Our proposed approach achieved mean intersection-over-union score of 89.5% on the DAVIS 2016 animal dataset [1] with only one labeled frame each video.

The contents of this repository are released under an [MIT](LICENSE) license.

![Overview](https://github.com/tengfeixue-victor/One-Shot-Animal-Video-Segmentation/blob/master/utils/overview.png?raw=true "Overview")

## Dependencies

The required python packages are listed in requirements.txt

## Overview

Our repository has three stages as described in paper: base training, objectness training, fine-tuning. Details will be introudced in the following.

## BubbleNet Selection
You can download the selection results in [bubbleNet_data.zip](https://drive.google.com/file/d/1mlOFxU0ueyt0CT7KX3lpd8NWbCh5JXtV/view?usp=sharing) (72 MB), unzip it in "datasets" folder. Noticeably, the BubbleNet that runs on the testing set does not need any labels. 

## Base Training
Download ImageNet pretrained model for XceptionNet [imagenet_pretrain_weights.zip](https://drive.google.com/file/d/1vqTu1X64tYsN224pA-LrEEbK1U3lv0Mp/view?usp=sharing) (585 MB), unzip it in "weights" folder

Download Pascal and its extension datasets [pascal_extension_dataset.zip](https://drive.google.com/file/d/16Ih-d3KPRmMrGUPpFO_QyW98Muu91LHs/view?usp=sharing) (1 GB), unzip it in "datasets" folder

Then you can run **train_pascal_base.py** to start base training. Noticeably, the dataset here is large, which requires you to have a GPU with 11GB memory or more. You can reduce the number of training images by delete items in "datasets/pretrain_benchmark_reduced.txt". 

"pretrain_benchmark_and_pascal.txt" represents the full dataset and "test_algorithm_pretrain_benchmark_reduced.txt" is used to debug fast.

You can also download our pre-trained weights on pascal and its extension [pascal_base_train_weights.zip](https://drive.google.com/file/d/1wQALrqcI3k9SVgWJ9A3wg_fZsWHfJ-Tu/view?usp=sharing) (121 MB), then unzip it in "weights" folder.

## Objectness Training
Download Davis 2016 training datasets [DAVIS2016_train_dataset.zip](https://drive.google.com/file/d/1KOrdPMZpFF3NK08cKpdtS99BpP_YGtif/view?usp=sharing) (254 MB), unzip it in "datasets" folder

Before running **train_davis_objectness.py**, you should ensure that pre-trained weights of pascal and its extension dataset are in "weights/pascal_base_train_weights" folder, and the iterations are matched with program here.

You can also download our pre-trained weights on DAVIS 2016 training set [objectness_weights.zip](https://drive.google.com/file/d/1fHbh-U_0G212u3iJIhv52O8NDvjG6hyH/view?usp=sharing) (121 MB), then unzip it in "weights" folder.

## Fine-tuning
Download DAVIS animal testing datasets (fine-tune and test) [finetune_test_dataset.zip](https://drive.google.com/file/d/1eNDbd3g2yg9zLs7RBk7Umi7BhTl04uuL/view?usp=sharing) (71 MB), unzip it in "datasets" folder

Similar to objectness training, please note that the pretrained weights in "weights/objectness_weights" should be matched with the one in program here. After running **finetune_test.py**, you can find the segmentation results in "results/segmentation" folder.

## Results Evaluation
If you wanna evaluate the results quantatively as we did in paper, you can refer to the DAVIS official code https://davischallenge.org/davis2016/code.html

## Citations
Complete citation is comming soon. The article is currently in [early access](https://ieeexplore.ieee.org/document/9556571).
```
@ARTICLE{9556571,  
  author={Xue, Tengfei and Qiao, Yongliang and Kong, He and Su, Daobilige and Pan, Shirui and Rafique, Khalid and Sukkarieh, Salah},  
  journal={IEEE Transactions on Industrial Informatics},   
  title={One-shot Learning-based Animal Video Segmentation},   
  year={2021},  
  volume={},  
  number={},  
  pages={1-1},  
  doi={10.1109/TII.2021.3117020}
}
```

## Reference
[1] Perazzi, F., Pont-Tuset, J., McWilliams, B., Van Gool, L., Gross, M. and Sorkine-Hornung, A., 2016. A benchmark dataset and evaluation methodology for video object segmentation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 724-732). https://davischallenge.org/davis2016/code.html
