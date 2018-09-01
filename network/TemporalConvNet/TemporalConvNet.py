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

    def buildNetwork(self, pretrained_input, pretrained_output):

        # self.input = tf.placeholder(shape = [None] + self._input_size, dtype = "float32") #change dtype for memory
        self.pretrained_input = pretrained_input
        self.input = pretrained_output

        num_time_steps = tf.shape(self.input)[0]
        TA_list = [tf.TensorArray(dtype = "float32", size = 0, dynamic_size = True, clear_after_read = False)] * self._num_layers
                   # clear_after_read = False if self._temporal_kernel_size[i] *
                   # self._temporal_dilation_factor[i] > self._temporal_stride[i] else True)
                   # for i in range(1, self._num_layers)]
        # TA_list.append(tf.TensorArray(dtype = "float32", size = 0, dynamic_size = True))

        TA_input = tf.TensorArray(dtype = "float32", size = 0, dynamic_size = True, clear_after_read = False).unstack(self.input)
                    # clear_after_read = False if self._temporal_kernel_size[0] *
                    # self._temporal_dilation_factor[0] > self._temporal_stride[0] else True).unstack(self.input)
        TA_list.insert(0, TA_input)

        def loopCond(iters, prev, curr):

            return iters < prev.size()

        def loopBody(iters, prev, curr):

            curr = tf.cond(tf.logical_and(tf.equal(tf.mod(tf.add(iters, 1), self._temporal_stride[self._curr_layer]), 0),
                    tf.greater(iters, tf.multiply(self._temporal_kernel_size[self._curr_layer],
                                                  self._temporal_dilation_factor[self._curr_layer]))),
                lambda: curr.write(curr.size(), self._convOp(prev, iters, self._temporal_kernel_size[self._curr_layer],
                                                             self._temporal_kernel_nums[self._curr_layer],
                                                             self._temporal_dilation_factor[self._curr_layer],
                                                             self._curr_layer)),
                lambda: curr
            )

            return iters + 1, prev, curr

        # final_iter, output, input, update = tf.while_loop(loopCond, loopBody, [0, TA_list, TA_input, [False] * (self._num_layers)], parallel_iterations=1)

        for self._curr_layer in range(self._num_layers):
            final_iter, TA_list[self._curr_layer], TA_list[self._curr_layer + 1] = tf.while_loop(loopCond, loopBody,
                                                                                         [0,
                                                                                          TA_list[self._curr_layer],
                                                                                          TA_list[self._curr_layer + 1]],
                                                                                                 parallel_iterations=1)

        self.output = [TA.stack() for TA in TA_list[1:]]
        self.labels, self.loss, self.train = self._lossOp(self.output)

    def initNetwork(self, model_path = None): #loading model not implemented
        init = tf.initialize_all_variables()
        self.sess = tf.InteractiveSession()
        self.sess.run(init)


    def predictNetwork(self, input):
        return self.sess.run(self.output, feed_dict={self.input: input})

    def trainWithFeed(self, input, label):
        loss, train, pretrained_output = self.sess.run([self.loss, self.train, self.input], feed_dict = {self.pretrained_input: input, self.labels: label})
        print(pretrained_output.shape)
        print(pretrained_output)
        return loss
