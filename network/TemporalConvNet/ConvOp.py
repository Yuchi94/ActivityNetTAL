import tensorflow as tf

"""
The TensorArray object (TA) is an Array of Tensors. Individual tensors can be extracted via
the read method. Tensors have dimensions defined as follows:
[Number of kernels, feature dimensions...]
Sequence of tensors are extracted and have dimensions defined as follows:
[Number of temporal steps, number of kernels, feature dimensions...]

Computation of the subsequent layer therefore sums over the first two dimensions of the preceding layer

"""

def conv1DOp(TA, iters, kernel_size, kernel_num, dilation, layer):

    #Get the dimensions of the tensors in the preceding layer
    shape = TA.read(0).get_shape().as_list()
    print(shape)

    if layer == 0:
        feature_dims = shape
        kernel_dims = 1
    else:
        feature_dims = shape[1:]
        kernel_dims = shape[0]

    kernel = [None] * kernel_num
    with tf.variable_scope("TCN_layer_" + str(layer), reuse = tf.AUTO_REUSE):
        for j in range(kernel_num):
            kernel[j] = tf.get_variable("kernel_" + str(j), shape = [kernel_size] + [kernel_dims] + [1 for i in range(len(feature_dims))])

    return tf.stack(                #stacking by number of kernels
        [tf.reduce_sum(             #summing over time (axis 0) and prev kernels (axis 1)
            tf.nn.relu(             #activation function
                tf.stack(           #constructing tensor from sliced reads over prev tensorarray
                    [TA.read(iters - i * dilation) for i in range(kernel_size)]) #reading tensor indices
            * tf.tile(kernel[j], [1, 1] + feature_dims)),   #tiling kernel to feature vector dims
            axis = [0, 1]) for j in range(kernel_num)])
