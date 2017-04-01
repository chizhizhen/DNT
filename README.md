# Dual Deep Network for Visual Tracking
## Introduction
DNT repository for **Dual Deep Network for Visual Tracking** is published in IEEE Transaction on Image Processing [[IEEE Xplore]](http://ieeexplore.ieee.org/document/7857085/) [[arXiv]](https://arxiv.org/abs/1612.06053). This package contains the source code to reproduce the experimental results of DNT paper. The source code is mainly written in MATLAB.

There a tracking benchmark tracking [repo](https://github.com/foolwood/benchmark_results). Check them out!

## Usage
+ Supported OS: the source code was tested on 64-bit Arch Linux OS, and it should also be executable in other linux distributions.

+ Dependencies:
  + Deep learning framework [caffe](http://caffe.berkeleyvision.org/) and all its dependencies.
  + Cuda-enabled GPUs.

+ Installation:

    + Install caffe: caffe is our customized version of the original caffe. Change directory into `./caffe` and compile the source code and the matlab interface following the [installation instruction of caffe](http://caffe.berkeleyvision.org/installation.html).

    + Download the 16-layer VGG network from [Simonyan's gist](https://gist.github.com/ksimonyan/211839e770f7b538e2d8), and put the caffemodel file under the `./feature_model` directory.

    + Run the demo code `run_tracker.m`. You can customize your own test sequences following the example inside.

## Having questions?
Feel free to contact us!

## Citing Our Work
If you find DNT useful in your research, please consider to cite our paper:

    @article{chi2017_tracking,
       title={Dual Deep Network for Visual Tracking},
       author={Chi, Zhizhen and Li, Hongyang and Lu, Huchuan and Yang, Minghsuan},
       volume={26},
       issue={4},
       pages={2005-2015},
       journal={IEEE Transaction on Image Processing},
       year={2017}
    }

