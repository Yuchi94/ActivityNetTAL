# ActivityNet Temporal Action Localization
This project is an entry into the Temporal Action Localization challenge for ActivityNet 2019. It seeks to leverage the long-term memory of [Temporal Convolutional Networks](https://arxiv.org/abs/1608.08242) (TCN). TCN style architectures have demonstrated effectiveness at capturing long-term dependencies in sequences that traditionally have been too difficult for RNN-based systems. A good example is Google [WaveNet](https://arxiv.org/pdf/1609.03499.pdf) where they managed to synthesize speech with a significantly more natural sound.
# What is Temporal Action Localization?
Temporal action localization is a challenge that Activity Net hosted in 2018 and will also host again in 2019. In this challenge, the goal is to localize specific action instances temporally within a untrimmed video sequence. As object recognition is more or less solved, the next big challenges in computer vision are spatial and temporal localization. You can read more about it [here](http://activity-net.org/challenges/2018/tasks/anet_localization.html).
# The TCN implementation
This is a work in progress. I have attempted to modularize the code for maintainability and organization. The TCN network lives in **network/TemporalConvNet**. It uses a tf while loop to dynamically adjust for the sequence length, thus reducing graph construction time and saving memory. Convolution operators and loss functions are user-defined functions - they can be easily "plugged" into the network to implement non-standard functions. Common convolutions and losses such as 1-D convolution and L2 loss are provided in **network/TemporalConvNet/ConvOp.py** and **network/TemporalConvNet/LossOp.py** respectively. 

Current usage of the network is as follows:

```python
import tensorflow as tf
from network.TemporalConvNet.TemporalConvNet import TemporalConvNet
from network.TemporalConvNet.ConvOp import *
from network.TemporalConvNet.LossOp import *

feature_shape = [1, 2048]

#The following params are sequenced by layer
dilation_factor = [2, 2, 4]
kernel_length = [10, 16, 20]
temporal_stride = [5, 7, 7]
num_kernels = [5, 3, 3]

#Build TCN
TCN = TemporalConvNet(feature_shape, 
                      dilation_factor,
                      kernel_length,
                      temporal_stride, 
                      num_kernels, 
                      conv1DOp, 
                      ClassificationCELoss(1e-4, 6))
input = tf.placeholder(tf.float32, [None, 2048])
TCN.buildNetwork(None, input)
TCN.initNetwork()

