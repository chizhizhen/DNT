## Dual Deep Network for Visual Tracking
# Introduction
DNT repository for Dual Deep Network for Visual Tracking is published in IEEE Transaction on Image Processing. This package contains the source code to reproduce the experimental results of DNT paper. The source code is mainly written in MATLAB.

# Usage
+ Supported OS: the source code was tested on 64-bit Arch Linux OS, and it should also be executable in other linux distributions.

+ Dependencies:
  + Deep learning framework [caffe](http://caffe.berkeleyvision.org/) and all its dependencies.
  + Cuda enabled GPUs.

+ Installation:
    i. Install caffe: caffe is our customized version of the original caffe. Change directory into ./caffe and compile the source code and the matlab interface following the [installation instruction of caffe](http://caffe.berkeleyvision.org/installation.html).

    ii. Download the 16-layer VGG network from [https://gist.github.com/ksimonyan/211839e770f7b538e2d8](https://gist.github.com/ksimonyan/211839e770f7b538e2d8), and put the caffemodel file under the ./feature_model directory.
    
    iii. Run the demo code run.m. You can customize your own test sequences following this example.

+ Citing Our Work
If you find DNT useful in your research, please consider to cite our paper:

    @inproceedings{ zhizhenchi,
       title={Dual Deep Network for Visual Tracking},
       author={Chi, Zhizhen and Li, Hongyang and Lu, Huchuan and Yang, Minghsuan},
       booktitle={IEEE Transaction on Image Processing},
       year={2017}
    }

+ License
The MIT License (MIT)

Copyright (c) 2017 Zhizhen Chi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.