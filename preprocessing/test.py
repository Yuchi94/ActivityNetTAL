import cv2
import tensorflow as tf
import numpy as np
import os
import gc
#Load all the files into memory

base_path = "dataset/Recognition_of_human_actions/"
categories = ["boxing", "handclapping", "handwaving", "jogging", "running", "walking"]

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v2

from slim.preprocessing import inception_preprocessing as ip

pretrained_input = tf.placeholder(tf.uint8, shape = (120, 160, 3))
image = ip.preprocess_image(pretrained_input, 299, 299)
with slim.arg_scope(resnet_v2.resnet_arg_scope()):
    net, endpoints = resnet_v2.resnet_v2_152(tf.expand_dims(image, axis = 0), is_training = False)

#load pretrained resnet
saver = tf.train.Saver()
sess = tf.InteractiveSession()
saver.restore(sess, 'pretrained/resnet_v2_152/resnet_v2_152.ckpt')

print("restored")

def getFramesFromVideo(video):
    cap = cv2.VideoCapture(video)
    #cap = cv2.VideoCapture('../dataset/Recognition_of_human_actions/boxing/person01_boxing_d1_uncomp.avi')
    frames = []
    while(cap.isOpened()):
        ret, frame = cap.read()
        if frame is None:
            break
        frame = frame.astype(np.uint8)

        # height, width = frame.shape[:2]
        # frame = cv2.resize(frame[:,20:-20,:], (0, 0), None, 224 / height, 224 / height, cv2.INTER_CUBIC)
        # height, width = frame.shape[:2]
        features = sess.run(net, feed_dict={pretrained_input: frame})
        # cv2.imshow("frame", frame)
        # cv2.waitKey(1)
        # frame = cv2.bilateralFilter(frame,3, 500,500)
        # frame = cv2.pyrDown(cv2.pyrDown(frame))
        frames.append(np.reshape(features, [-1]))

    cap.release()
    return frames

for cat in categories:
    dataset = []
    for file in os.listdir(base_path + cat):
        filename = os.fsdecode(file)
        frames = getFramesFromVideo(os.path.join(base_path + cat, filename))
        features = np.stack(frames)
        dataset.append(features)
        print(filename)
        print(features)

    np.savez(cat + ".npz", *dataset)


