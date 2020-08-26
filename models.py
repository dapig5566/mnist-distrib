import tensorflow.compat.v1 as tf


class Classifier:
    def __init__(self, num_classes, num_channels=64):
        self.num_channels = num_channels
        self.num_classes = num_classes

    def build_convnet(self, input, reuse=tf.AUTO_REUSE, training=True):
        with tf.variable_scope("convnet", reuse=reuse):

            x = tf.layers.conv2d(input, self.num_channels, 4, 2, "same")
            x = tf.nn.leaky_relu(x)

            x = tf.layers.conv2d(x, 2*self.num_channels, 4, 2, "same", use_bias=False)
            x = tf.layers.batch_normalization(x, training=training)
            x = tf.nn.leaky_relu(x)
            x = tf.reshape(x, [-1, 7*7*2*self.num_channels])
            x = tf.layers.dense(x, 1024, use_bias=False)
            x = tf.layers.batch_normalization(x, training=training)
            x = tf.nn.leaky_relu(x)

            logits = tf.layers.dense(x, self.num_classes)

        return logits