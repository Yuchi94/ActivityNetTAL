#Test script
import tensorflow as tf
import numpy as np
import cv2

from TemporalConvNet import TemporalConvNet

#define convOp
def convOp(TA, iters, kernel, dilation):
    #return TA.read(iters)
    print(TA.read(0).shape)
    return tf.add_n([TA.read(iters - i * dilation) for i in range(kernel)])/kernel

    return tf.reduce_mean(tf.stack([TA.read(iters - i * dilation) for i in range(kernel)]))

#build model
net = TemporalConvNet([90, 160, 3], [1, 2], [3, 2], [5, 10], convOp, None)
net.buildNetwork()
net.initNetwork()

#load video frames
cap = cv2.VideoCapture('videoplayback.mp4')
frames = []
while(cap.isOpened()):
    ret, frame = cap.read()
             
    if frame is None: 
        break
    frame = cv2.bilateralFilter(frame,3, 500,500)  
    frame = cv2.pyrDown(cv2.pyrDown(frame))
    frames.append(frame)

    
cap.release()
    
print(len(frames)) 
input = np.stack(frames)

output = net.predict(input)
test_output = output[1]
print(output[0].shape)
print(output[1].shape)
for i in range(test_output.shape[0]):
    #print(test_output[i,:,:,0])
    cv2.imshow("output", test_output[i,:,:,:] / 255)
    cv2.waitKey(1)




