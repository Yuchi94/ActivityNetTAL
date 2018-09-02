from network.TemporalConvNet.TemporalConvNet import TemporalConvNet
from network.TemporalConvNet.ConvOp import *
from network.TemporalConvNet.LossOp import *

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v2
import numpy as np

import random
import gc

USE_LIVE_CNN = False

#Load the files into memory
boxing = np.load("preprocessing/boxing.npz")
handclapping = np.load("preprocessing/handclapping.npz")
handwaving = np.load("preprocessing/handwaving.npz")
running = np.load("preprocessing/running.npz")
jogging = np.load("preprocessing/jogging.npz")
walking = np.load("preprocessing/walking.npz")


def getOneHot(label):
    if label == 0:
        return np.array([[1, 0, 0, 0, 0, 0]])
    if label == 1:
        return np.array([[0, 1, 0, 0, 0, 0]])
    if label == 2:
        return np.array([[0, 0, 1, 0, 0, 0]])
    if label == 3:
        return np.array([[0, 0, 0, 1, 0, 0]])
    if label == 4:
        return np.array([[0, 0, 0, 0, 1, 0]])
    if label == 5:
        return np.array([[0, 0, 0, 0, 0, 1]])

def getVideo(label):
    if label == 0:
        return boxing[random.choice(boxing.files)]
    if label == 1:
        return handclapping[random.choice(handclapping.files)]
    if label == 2:
        return handwaving[random.choice(handwaving.files)]
    if label == 3:
        return running[random.choice(running.files)]
    if label == 4:
        return jogging[random.choice(jogging.files)]
    if label == 5:
        return walking[random.choice(walking.files)]




if USE_LIVE_CNN:
    #initialize pretrained resnet
    pretrained_input = tf.placeholder(tf.float32, shape = (None, 224, 224, 3))
    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        net, endpoints = resnet_v2.resnet_v2_152(pretrained_input, is_training = False)

    pretrained_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

    print ("Pretrained network initialized")
    #build temporal conv net
    #input_size, temporal_dilation_factor, temporal_kernel_size, temporal_stride, temporal_kernel_nums,  convOp, lossOp
    TCN = TemporalConvNet([1, 2048], [1], [5], [1], [1], conv1DOp, ClassificationCELoss(1e-3, 6))
    TCN.buildNetwork(pretrained_input, tf.reshape(net, [-1, 2048]))
    print ("TCN built")
    TCN.initNetwork()
    print ("TCN initialized")

    #load pretrained resnet
    saver = tf.train.Saver(pretrained_variables)
    with tf.Session() as sess:
        saver.restore(sess, 'pretrained/resnet_v2_152/resnet_v2_152.ckpt')
    print ("Pretrained network initialized")

else:

    # build temporal conv net
    # input_size, temporal_dilation_factor, temporal_kernel_size, temporal_stride, temporal_kernel_nums,  convOp, lossOp
    TCN = TemporalConvNet([1, 2048], [10], [3], [1], [1], conv1DOp, ClassificationCELoss(1e-4, 6))
    input = tf.placeholder(tf.float32, [None, 2048])
    TCN.buildNetwork(None, input)
    print("TCN built")
    TCN.initNetwork()
    print("TCN initialized")
    # try training the network
    for i in range(1000):
        label = random.randint(0, 5)
        video = getVideo(label).astype(np.float)
        onehot = getOneHot(label)
        # print(video)
        loss = TCN.trainWithFeed(video, onehot)
        print(loss)




