"""
File to run the main training script for the KTH dataset.
Requires the video files to be preprocessed into feature vectors.
"""

from network.TemporalConvNet.TemporalConvNet import TemporalConvNet
from network.TemporalConvNet.ConvOp import *
from network.TemporalConvNet.LossOp import *

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v2
import numpy as np

import random

USE_LIVE_CNN = False

#Load the files into memory
load_path = "preprocessing/KTH/"
boxing = np.load(load_path + "boxing.npz")
handclapping = np.load(load_path + "handclapping.npz")
handwaving = np.load(load_path + "handwaving.npz")
running = np.load(load_path + "running.npz")
jogging = np.load(load_path + "jogging.npz")
walking = np.load(load_path + "walking.npz")


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




if USE_LIVE_CNN: #Currently not used
    pretrained_input = tf.placeholder(tf.float32, shape = (None, 224, 224, 3))
    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        net, endpoints = resnet_v2.resnet_v2_152(pretrained_input, is_training = False)

    pretrained_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

    #Build TCN
    TCN = TemporalConvNet([1, 2048], [1,1], [5, 3], [1, 1], [1, 1], conv1DOp, ClassificationCELoss(1e-3, 6))
    TCN.buildNetwork(pretrained_input, tf.reshape(net, [-1, 2048]))
    TCN.initNetwork()

    #Load pretrained resnet
    saver = tf.train.Saver(pretrained_variables)
    with tf.Session() as sess:
        saver.restore(sess, 'pretrained/resnet_v2_152/resnet_v2_152.ckpt')

else: #Use static preprocessed inputs
    TCN = TemporalConvNet([1, 2048], [10], [3], [1], [1], conv1DOp, ClassificationCELoss(1e-4, 6))
    input = tf.placeholder(tf.float32, [None, 2048])
    TCN.buildNetwork(None, input)
    TCN.initNetwork()

    # Train the network
    for i in range(20000):
        label = random.randint(0, 5)
        video = getVideo(label).astype(np.float)
        onehot = getOneHot(label)
        loss = TCN.trainWithFeed(video, onehot)
        print(loss)




