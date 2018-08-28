import tensorflow as tf

def conv1DOp(TA, iters, kernel_size, dilation, layer):
    shape = TA.read(0).get_shape().as_list()
    print(shape)

    with tf.variable_scope("TCN_layer_" + str(layer), reuse = tf.AUTO_REUSE):
        kernel = tf.get_variable("kernel", shape = [kernel_size] + [1 for i in range(len(shape))])

    return tf.reduce_sum(tf.stack([TA.read(iters - i * dilation) for i in range(kernel_size)])
            * tf.tile(kernel, [1] + shape), axis = 0)
