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
TRAIN = True 
EVAL = True
#Load the files into memory
load_path = "preprocessing/KTH/"
save_path = "saved_models/KTH/"

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
        return boxing[random.choice(boxing.files[:-20])]
    if label == 1:
        return handclapping[random.choice(handclapping.files[:-20])]
    if label == 2:
        return handwaving[random.choice(handwaving.files[:-20])]
    if label == 3:
        return running[random.choice(running.files[:-20])]
    if label == 4:
        return jogging[random.choice(jogging.files[:-20])]
    if label == 5:
        return walking[random.choice(walking.files[:-20])]



if TRAIN:
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
        TCN = TemporalConvNet([1, 2048], [2,2], [10,10], [2,2], [4,4], conv1DOp, ClassificationCELoss(1e-4, 6))
        input = tf.placeholder(tf.float32, [None, 2048])
        TCN.buildNetwork(None, input)
        TCN.initNetwork()
        
        print("training")
        # Train the network
        for i in range(20000):
            label = random.randint(0, 5)
            video = getVideo(label).astype(np.float)
            onehot = getOneHot(label)
            print(video.shape)
            loss = TCN.trainWithFeed(video, onehot)
            print(loss)
        
        TCN.saveNetwork(save_path)


if EVAL:
    TCN = TemporalConvNet([1, 2048], [2,2], [10,10], [2,2], [4,4], conv1DOp, ClassificationCELoss(1e-4, 6))
    input = tf.placeholder(tf.float32, [None, 2048])
    TCN.buildNetwork(None, input)
    TCN.initNetwork(save_path)

    correct = 0
    incorrect = 0
    for vid in [boxing[key] for key in boxing.files[-20:]]:
        pred = np.mean(TCN.predictNetwork(vid), axis = 0)
        if np.argmax(pred) == 0:
            correct += 1
        else:
            incorrect += 1
    print("Incorrect: " + str(incorrect))
    print("Correct: " + str(correct))

    correct = 0
    incorrect = 0
    for vid in [handclapping[key] for key in handclapping.files[-20:]]:
        pred = np.mean(TCN.predictNetwork(vid), axis = 0)
        if np.argmax(pred) == 1:
            correct += 1
        else:
            incorrect += 1
    print("Incorrect: " + str(incorrect))
    print("Correct: " + str(correct))

    correct = 0
    incorrect = 0
    for vid in [handwaving[key] for key in handwaving.files[-20:]]:
        pred = np.mean(TCN.predictNetwork(vid), axis = 0)
        if np.argmax(pred) == 2:
            correct += 1
        else:
            incorrect += 1
    print("Incorrect: " + str(incorrect))
    print("Correct: " + str(correct))

    correct = 0
    incorrect = 0
    for vid in [running[key] for key in running.files[-20:]]:
        pred = np.mean(TCN.predictNetwork(vid), axis = 0)
        if np.argmax(pred) == 3:
            correct += 1
        else:
            incorrect += 1
    print("Incorrect: " + str(incorrect))
    print("Correct: " + str(correct))

    correct = 0
    incorrect = 0
    for vid in [jogging[key] for key in jogging.files[-20:]]:
        pred = np.mean(TCN.predictNetwork(vid), axis = 0)
        if np.argmax(pred) == 4:
            correct += 1
        else:
            incorrect += 1
    print("Incorrect: " + str(incorrect))
    print("Correct: " + str(correct))

    correct = 0
    incorrect = 0
    for vid in [walking[key] for key in walking.files[-20:]]:
        pred = np.mean(TCN.predictNetwork(vid), axis = 0)
        if np.argmax(pred) == 5:
            correct += 1
        else:
            incorrect += 1
    print("Incorrect: " + str(incorrect))
    print("Correct: " + str(correct))


