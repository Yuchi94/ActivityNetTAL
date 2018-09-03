import tensorflow as tf
import numpy as np

class TemporalConvNet:

    def __init__(self, input_size, temporal_dilation_factor, temporal_kernel_size, temporal_stride, temporal_kernel_nums,  convOp, lossOp):

        if not (len(temporal_dilation_factor) == len(temporal_stride) == len(temporal_kernel_size)):
            raise ValueError("Length of dilation factors, kernel sizes, and strides must match")

        self._input_size = input_size
        self._temporal_dilation_factor = temporal_dilation_factor
        self._temporal_kernel_size = temporal_kernel_size
        self._temporal_stride = temporal_stride
        self._temporal_kernel_nums = temporal_kernel_nums
        self._convOp = convOp
        self._lossOp = lossOp
        self._num_layers = len(temporal_stride)
        self._curr_layer = None

        self.sess = None

        ###DEBUG###
        self.probe = None

    def buildNetwork(self, pretrained_input, pretrained_output):

        self.pretrained_input = pretrained_input
        self.input = pretrained_output

        TA_list = [tf.TensorArray(dtype = "float32", size = 0, dynamic_size = True, clear_after_read = False) for i in range(self._num_layers)]
        TA_input = tf.TensorArray(dtype = "float32", size = 0, dynamic_size = True, clear_after_read = False).unstack(self.input)
        TA_list.insert(0, TA_input)

        def loopCond(iters, prev, curr):
            #equation to determine the start index of the current layer is:
            #(kernel_size - 1) * dilation_factor + iters * stride

            return (self._temporal_kernel_size[self._curr_layer] - 1) * \
                   self._temporal_dilation_factor[self._curr_layer] + \
                    self._temporal_stride[self._curr_layer] * iters < prev.size()

        def loopBody(iters, prev, curr):

            index = (self._temporal_kernel_size[self._curr_layer] - 1) * \
                   self._temporal_dilation_factor[self._curr_layer] + \
                    self._temporal_stride[self._curr_layer] * iters

            curr = curr.write(iters, self._convOp(prev, index, self._temporal_kernel_size[self._curr_layer],
                                                             self._temporal_kernel_nums[self._curr_layer],
                                                             self._temporal_dilation_factor[self._curr_layer],
                                                             self._curr_layer))

            return iters + 1, prev, curr

        for self._curr_layer in range(self._num_layers):
            final_iter, TA_list[self._curr_layer], TA_list[self._curr_layer + 1] = tf.while_loop(loopCond,
                                                                                        loopBody,
                                                                                        [0,
                                                                                        TA_list[self._curr_layer],
                                                                                        TA_list[self._curr_layer + 1]])

        self.output = [TA.stack() for TA in TA_list[1:]]
        self.labels, self.loss, self.train, self.probe = self._lossOp(self.output)

    def initNetwork(self, model_path = None): #loading model not implemented
        init = tf.initialize_all_variables()
        self.sess = tf.InteractiveSession()
        self.sess.run(init)


    def predictNetwork(self, input):
        return self.sess.run(self.output, feed_dict={self.input: input})

    def trainWithFeed(self, input, label):
        # print(input)
        if self.pretrained_input:
            loss, train, pretrained_output, probe = self.sess.run([self.loss,
                                                                   self.train,
                                                                   self.input,
                                                                   self.probe],
                                                                  feed_dict = {self.pretrained_input: input,
                                                                               self.labels: label})
        else:
            loss, train, pretrained_output, probe = self.sess.run([self.loss,
                                                                   self.train,
                                                                   self.input,
                                                                   self.probe],
                                                                  feed_dict = {self.input: input,
                                                                               self.labels: label})

        # print(probe)
        # print(pretrained_output.shape)
        # print(pretrained_output)
        return loss
