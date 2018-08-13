import tensorflow as tf
import numpy as np

class TemporalConvNet:
    
    def __init__(self, input_size, temporal_dilation_factor, temporal_kernel_size, temporal_stride, convOp):
        
        if len(temporal_dilation_factor) != len(temporal_stride) != len(temporal_kernel_size):
            raise ValueError("Length of dilation factors, kernel sizes, and strides must match")

        self._input_size = input_size
        self._temporal_dilation_factor = temporal_dilation_factor
        self._temporal_kernel_size = temporal_kernel_size
        self._temporal_stride = temporal_stride
        self._convOp = convOp
        self._num_layers = len(temporal_stride)
        
        self.sess = None
    
    def buildNetwork(self):
        
        self.input = tf.placeholder(shape = [None] + self._input_size, dtype = "float32") #change dtype for memory
        num_time_steps = tf.shape(self.input)[0]
        TA_list = [tf.TensorArray(dtype = "float32", size = 0, dynamic_size = True)] * self._num_layers
        TA_input = tf.TensorArray(dtype = "float32", size = 0, dynamic_size = True).unstack(self.input)

        def loopCond(iters, *_):
            return iters < num_time_steps

        def loopBody(iters, TA, input):

            #TA[0] = TA[0].write(iters, input.read(iters))
            #return iters + 1, TA, input
            print(iters)
            print("!")
            TA[0] = tf.cond(tf.logical_and(tf.equal(tf.mod(tf.add(iters, 1), self._temporal_stride[0]), 0), tf.greater(iters, tf.multiply(self._temporal_kernel_size[0], self._temporal_dilation_factor[0]))),
                lambda: TA[0].write(iters, self._convOp(input, iters, self._temporal_kernel_size[0], self._temporal_dilation_factor[0])),
                lambda: TA[0]
            )
         #   if (iters + 1) % (self._temporal_stride[0]) == 0 and iters > (self._temporal_kernel_size[0] * self._temporal_dilation_factor[0]):
         #       TA[0] = TA[0].write(iters, self._convOp(input, iters, self._temporal_kernel_size[0], self._temporal_dilation_factor[0]))
         #   else:
         #       return iters + 1, TA, input

            for i in range(1, self._num_layers):
                TA[i] = tf.cond(tf.logical_and(tf.equal(tf.mod(TA[i-1].size(), self._temporal_stride[i]), 0), tf.greater_equal(TA[i-1].size(), tf.multiply(self._temporal_kernel_size[i], self._temporal_dilation_factor[i]))),
                    lambda: TA[i].write(iters, self._convOp(input, iters, self._temporal_kernel_size[i], self._temporal_dilation_factor[i])),
                    lambda: TA[i]
                )
               # if (TA[i-1].size()) % (self._temporal_stride[i]) == 0 and TA[i-1].size() >= (self._temporal_kernel_size[i] * self._temporal_dilation_factor[i]):
               #     TA[i] = TA[i].write(iters, self._convOp(TA[i-1], TA[i-1].size() - 1, self._temporal_kernel_size[i], self._temporal_dilation_factor[i]))
               # else:
               #     return iters + 1, TA, input
       #     base_mult = 1

       #     for i, (dilation, kernel, stride)  in enumerate(zip(self._temporal_dilation_factor, self._temporal_kernel_size, self._temporal_stride)):
       #         if 
       #         if iters % (stride * base_mult) == 0 and iters > (kernel * dilation * base_mult):
       #             TA[i + 1] = TA[i + 1].write(iters, self._convOp(TA[i], iters, kernel, dilation))
       #             base_mult = base_mult * stride

       #         else:
       #             break

            return iters + 1, TA, input
       # 
            
        final_iter, output, _ = tf.while_loop(loopCond, loopBody, [0, TA_list, TA_input])
        self.output = [TA.stack() for TA in output]

    def initNetwork(self, model_path = None): #loading model not implemented
        init = tf.initialize_all_variables()
        self.sess = tf.InteractiveSession()
        self.sess.run(init)


    def predict(self, input):
        return self.sess.run(self.output, feed_dict={self.input: input})        

    def train(self, input, label):
        pass
            
