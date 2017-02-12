import skimage.io  # bug. need to import this before tensorflow
import skimage.transform  # bug. need to import this before tensorflow
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages
from config import Config
import tensorflow as tf
import datetime
import numpy as np
from ops import *


#tf.app.flags.DEFINE_integer('input_size', 224, "input image size")


activation = tf.nn.relu


class RESNET(object):
    def __init__(self, sess,
                num_classes=1000,
                num_blocks=[3, 4, 6, 3],  # defaults to 50-layer network
                use_bias=False, # defaults to using batch norm
                bottleneck=True,
                is_training=True):
        self.c = Config()
        self.sess = sess
        self.num_classes = num_classes
        self.num_blocks = num_blocks
        self.c['bottleneck'] = bottleneck
        self.c['use_bias'] = use_bias
        self.c['fc_units_out'] = num_classes
        self.c['is_training'] = tf.convert_to_tensor(is_training,
                                            dtype='bool',
                                            name='is_training')

    def loss(self, logits, labels):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
     
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

        loss_ = tf.add_n([cross_entropy_mean] + regularization_losses)
        tf.scalar_summary('loss', loss_)

        return loss_
    def inference(self, x):
        self.c['ksize'] = 3
        self.c['stride'] = 1
        self.c['stack_stride'] = 2

        with tf.variable_scope('scale1'):
            self.c['conv_filters_out'] = 64
            self.c['ksize'] = 7
            self.c['stride'] = 2
            x = conv(x, self.c)
            x = bn(x, self.c)
            x = activation(x)

        with tf.variable_scope('scale2'):
            x = max_pool(x, ksize=3, stride=2)
            self.c['num_blocks'] = self.num_blocks[0]
            self.c['stack_stride'] = 1
            self.c['block_filters_internal'] = 64
            x = stack(x, self.c)

        with tf.variable_scope('scale3'):
            self.c['num_blocks'] = self.num_blocks[1]
            self.c['block_filters_internal'] = 128
            assert self.c['stack_stride'] == 2
            x = stack(x, self.c)

        with tf.variable_scope('scale4'):
            self.c['num_blocks'] = self.num_blocks[2]
            self.c['block_filters_internal'] = 256
            x = stack(x, self.c)

        with tf.variable_scope('scale5'):
            self.c['num_blocks'] = self.num_blocks[3]
            self.c['block_filters_internal'] = 512
            x = stack(x, self.c)

        # post-net
        x = tf.reduce_mean(x, reduction_indices=[1, 2], name="avg_pool")

        if self.num_classes != None:
            with tf.variable_scope('fc'):
                x = fc(x, self.c)

        return x


    # This is what they use for CIFAR-10 and 100.
    # See Section 4.2 in http://arxiv.org/abs/1512.03385
    def inference_small(self, x,
                        is_training,
                        num_blocks=3, # 6n+2 total weight layers will be used.
                        use_bias=False, # defaults to using batch norm
                        num_classes=10):
        self.c['num_classes'] = num_classes
        self.c['num_blocks'] = num_blocks
        _inference_small_config(x, self.c)

    def _inference_small_config(self, x):
        self.c['bottleneck'] = False
        self.c['ksize'] = 3
        self.c['stride'] = 1
        with tf.variable_scope('scale1'):
            self.c['conv_filters_out'] = 16
            self.c['block_filters_internal'] = 16
            self.c['stack_stride'] = 1
            x = conv(x, self.c)
            x = bn(x, self.c)
            x = activation(x)
            x = stack(x, self.c)

        with tf.variable_scope('scale2'):
            self.c['block_filters_internal'] = 32
            self.c['stack_stride'] = 2
            x = stack(x, self.c)

        with tf.variable_scope('scale3'):
            self.c['block_filters_internal'] = 64
            self.c['stack_stride'] = 2
            x = stack(x, self.c)

        # post-net
        x = tf.reduce_mean(x, reduction_indices=[1, 2], name="avg_pool")

        if self.c['num_classes'] != None:
            with tf.variable_scope('fc'):
                x = fc(x, self.c)

        return x


