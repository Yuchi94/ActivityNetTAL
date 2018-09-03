import tensorflow as tf

"""
Output is a list of stacked tensors, each corresponding to a layer in the TCN. Thus they have dimensions as follows:
[Number of temporal steps, number of kernels, feature dimensions...]

"""

def L2Loss(learning_rate):
    """
    Standard L2 regression loss
    """

    def _loss(output):
        labels = [tf.placeholder(tf.float32, shape = output[i].shape) for i in range(len(output))]
        loss = [tf.nn.l2_loss(labels[i] - output[i]) if labels[i] is not None else 0 for i in range(len(output))]
        train = tf.train.AdamOptimizer(learning_rate).minimize(tf.add_n(loss))

        return labels, loss, train

    return _loss


def ClassificationCELoss(learning_rate, classes):
    """
    This loss only cares about the upper most layer and will assign all labels in that layer to the class
    of the video
    """
    def map_function(tensor):
        return tf.reshape(tf.layers.dense(tf.reshape(tensor, [1, -1]),
                                          classes,
                                          name = "Final_FC_Layer",
                                          reuse = tf.AUTO_REUSE),
                          [-1])


    def _loss(output):
        logits = tf.map_fn(map_function, output[-1])
        labels = tf.placeholder(tf.float32, shape = [1, classes])
        probe = tf.nn.softmax(logits)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = tf.ones_like(logits) * labels, logits = logits))
        train = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        return labels, loss, train, probe

    return _loss
