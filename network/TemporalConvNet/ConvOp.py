import tensorflow as tf

def conv1DOp(TA, iters, kernel_size, dilation, layer):
    shape = TA.read(0).shape

    with tf.variable_scope("TCN_layer_" + layer, reuse = tf.AUTO_REUSE):
        kernel = tf.get_variable("kernel", shape = [kernel_size] + shape) 
    
    return tf.reduce_sum(tf.stack([TA.read(iters - i * dilation) for i in range(kernel)]) 
            * tf.tile(kernel, [1] + shape), axis = 0) 
