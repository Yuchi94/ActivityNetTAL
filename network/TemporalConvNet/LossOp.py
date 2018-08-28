#output is a list of tensors corresponding to the layers
import tensorflow as tf

def L2Loss(learning_rate):

    def _L2Loss(output):
        labels = [tf.placeholder(tf.float32, shape = output[i].shape) for i in range(len(output))]
        loss = [tf.nn.l2_loss(labels[i] - output[i]) if labels[i] is not None else 0 for i in range(len(output))]
        train = tf.train.AdamOptimizer(learning_rate).minimize(tf.add_n(loss))

        return labels, loss, train

    return _L2Loss
