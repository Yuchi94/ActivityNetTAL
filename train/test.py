from network.TemporalConvNet.TemporalConvNet import TemporalConvNet
from network.TemporalConvNet.ConvOp import *
from network.TemporalConvNet.LossOp import *

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v2
import numpy as np

#initialize pretrained resnet
pretrained_input = tf.placeholder(tf.float32, shape = (None, 160, 90, 3))
with slim.arg_scope(resnet_v2.resnet_arg_scope()):
    net, endpoints = resnet_v2.resnet_v2_152(pretrained_input)

pretrained_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
TCN = TemporalConvNet([2048], [1,1], [3,3], [5,5], conv1DOp, L2Loss(1e-3))
TCN.buildNetwork()
saver = tf.train.Saver(pretrained_variables)
with tf.Session() as sess:
    saver.restore(sess, 'pretrained/resnet_v2_152/resnet_v2_152.ckpt')

