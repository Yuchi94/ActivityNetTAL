import tensorflow as tf
import numpy as np

class TemporalConvNet:

    def __init__(self, input_size, temporal_dilation_factor, temporal_kernel_size, temporal_stride, convOp, lossOp):

        if not (len(temporal_dilation_factor) == len(temporal_stride) == len(temporal_kernel_size)):
            raise ValueError("Length of dilation factors, kernel sizes, and strides must match")

        self._input_size = input_size
        self._temporal_dilation_factor = temporal_dilation_factor
        self._temporal_kernel_size = temporal_kernel_size
        self._temporal_stride = temporal_stride
        self._convOp = convOp
        self._lossOp = lossOp
        self._num_layers = len(temporal_stride)

        self.sess = None

    def buildNetwork(self):

        self.input = tf.placeholder(shape = [None] + self._input_size, dtype = "float32") #change dtype for memory
        num_time_steps = tf.shape(self.input)[0]
        TA_list = [tf.TensorArray(dtype = "float32", size = 0, dynamic_size = True, clear_after_read = False)]
                   # clear_after_read = False if self._temporal_kernel_size[i] *
                   # self._temporal_dilation_factor[i] > self._temporal_stride[i] else True)
                   # for i in range(1, self._num_layers)]
        TA_list.append(tf.TensorArray(dtype = "float32", size = 0, dynamic_size = True))

        TA_input = tf.TensorArray(dtype = "float32", size = 0, dynamic_size = True,
                    clear_after_read = False if self._temporal_kernel_size[0] *
                    self._temporal_dilation_factor[0] > self._temporal_stride[0] else True).unstack(self.input)


        def loopCond(iters, *_):
            return iters < num_time_steps

        def loopBody(iters, TA, input, update):

            TA[0] = tf.cond(tf.logical_and(tf.equal(tf.mod(tf.add(iters, 1), self._temporal_stride[0]), 0),
                    tf.greater(iters, tf.multiply(self._temporal_kernel_size[0], self._temporal_dilation_factor[0]))),
                lambda: TA[0].write(TA[0].size(), self._convOp(input, iters, self._temporal_kernel_size[0], self._temporal_dilation_factor[0], 0)),
                lambda: TA[0]
            )

            update[0] = tf.cond(tf.logical_and(tf.equal(tf.mod(tf.add(iters, 1), self._temporal_stride[0]), 0),
                    tf.greater(iters, tf.multiply(self._temporal_kernel_size[0], self._temporal_dilation_factor[0]))),
                lambda: True,
                lambda: False
            )


            for i in range(1, self._num_layers):
                TA[i] = tf.cond(tf.logical_and(tf.logical_and(tf.equal(tf.mod(TA[i-1].size(), self._temporal_stride[i]), 0),
                    tf.greater_equal(TA[i-1].size(), tf.multiply(self._temporal_kernel_size[i], self._temporal_dilation_factor[i]))),
                    update[i-1]),
                    lambda: TA[i].write(TA[i].size(), self._convOp(TA[i-1], TA[i-1].size() - 1, self._temporal_kernel_size[i], self._temporal_dilation_factor[i], i)),
                    lambda: TA[i]
                )

                update[i] = tf.cond(tf.logical_and(tf.logical_and(tf.equal(tf.mod(TA[i-1].size(), self._temporal_stride[i]), 0),
                    tf.greater_equal(TA[i-1].size(), tf.multiply(self._temporal_kernel_size[i], self._temporal_dilation_factor[i]))),
                    update[i-1]),
                    lambda: True,
                    lambda: False
                )

                update[i - 1] = tf.cond(tf.logical_and(tf.logical_and(tf.equal(tf.mod(TA[i-1].size(), self._temporal_stride[i]), 0),
                    tf.greater_equal(TA[i-1].size(), tf.multiply(self._temporal_kernel_size[i], self._temporal_dilation_factor[i]))),
                    update[i-1]),
                    lambda: False,
                    lambda: update[i-1]
                )

            return iters + 1, TA, input, update

        final_iter, output, input, update = tf.while_loop(loopCond, loopBody, [0, TA_list, TA_input, [False] * (self._num_layers)])
        self.output = [TA.stack() for TA in output]
        self.labels, self.loss, self.train = self._lossOp(self.output)

    def initNetwork(self, model_path = None): #loading model not implemented
        init = tf.initialize_all_variables()
        self.sess = tf.InteractiveSession()
        self.sess.run(init)


    def predict(self, input):
        return self.sess.run(self.output, feed_dict={self.input: input})

    def train(self, input, label):
        loss, train = self.sess.run([self.loss, self.train], feed_dict = {self.input: input, self.label: label})
        return loss
