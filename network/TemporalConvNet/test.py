#Test script
import tensorflow as tf
import numpy as np
import cv2

from TemporalConvNet import TemporalConvNet

#define convOp
def convOp(TA, iters, kernel, dilation):
    return TA.read(iters)

#build model
net = TemporalConvNet([90, 160, 3], [1], [3], [1], convOp)
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
    
    
input = np.stack(frames)

test_output = net.predict(input)[0]
for i in range(test_output.shape[0]):
    print(test_output[i,:,:,0])
    cv2.imshow("output", test_output[i,:,:,:] / 255)
    cv2.waitKey(1)




