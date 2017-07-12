# -*- coding" utf-8 -*-

# This module is a simple CNN with 2 convolution layer,
# 2 full connected layer and 1 output layer.
# Batch nornalization is not used in the simle CNN.
import os
import tensorflow as tf
import sys
import re

import ins_image_input
BATCH_SIZE = 50
CLASSES_NUM = 12
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 41481
NUM_EXAMPLES_PER_EPOCH_FOR_TEST = 13826
NUM_EPOCHS_PER_DECAY = 350
MOVING_AVERAGE_DECAY = 0.9999
LEARNING_RATE_DECAY_FACTOR = 0.1
INITIAL_LEARNING_RATE = 0.1

def _activation_summary(x):
    tensor_name = x.op.name
    tf.summary.histogram(tensor_name + '/activations',x)
    tf.summary.scalar(tensor_name + '/sparsity',tf.nn.zero_fraction(x))

def _variable_with_weight_decay(name,shape,wd,stddev):
    var = tf.get_variable(name,shape,initializer=tf.truncated_normal_initializer(stddev=stddev))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var),wd,name='weight_loss')
        tf.add_to_collection('loss',weight_decay)
    return var

def inference(images):
    with tf.variable_scope('conv1') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[5, 5, 3, 64],
                                             stddev=5e-2,
                                             wd=0.0)
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', [64], initializer=tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv1)

    # pool1
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool1')
    # norm1
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm1')

    # conv2
    with tf.variable_scope('conv2') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[5, 5, 64, 64],
                                             stddev=5e-2,
                                             wd=0.0)
        conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', [64], initializer=tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv2)

    # norm2
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm2')
    # pool2
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    # local3
    with tf.variable_scope('local3') as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        reshape = tf.reshape(pool2, [BATCH_SIZE, -1])
        dim = reshape.get_shape()[1].value
        weights = _variable_with_weight_decay('weights', shape=[dim, 384],
                                              stddev=0.04, wd=0.004)
        biases =tf.get_variable('biases', [384], initializer=tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
        _activation_summary(local3)

    # local4
    with tf.variable_scope('local4') as scope:
        weights = _variable_with_weight_decay('weights', shape=[384, 192],
                                              stddev=0.04, wd=0.004)
        biases =tf.get_variable('biases', [192], initializer=tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
        _activation_summary(local4)

    # linear layer(WX + b),
    # We don't apply softmax here because
    # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
    # and performs the softmax internally for efficiency.
    with tf.variable_scope('softmax_linear') as scope:
        weights = _variable_with_weight_decay('weights', [192, CLASSES_NUM],
                                              stddev=1 / 192.0, wd=0.0)
        biases = tf.get_variable('biases', [CLASSES_NUM], initializer=tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
        _activation_summary(softmax_linear)

    return softmax_linear

def loss(logits,labels):
    labels = tf.cast(labels,tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits,
                                                                   name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy,name='cross_entropy')
    tf.add_to_collection('losses',cross_entropy_mean)

    return tf.add_n(tf.get_collection('losses'),name='total_loss')

def _add_loss_summaries(total_loss):
    loss_averages = tf.train.ExponentialMovingAverage(0.9,name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    for l in losses+[total_loss]:
        tf.summary.scalar(l.op.name + ' (raw)',l)
        tf.summary.scalar(l.op.name,loss_averages.average(l))

    return loss_averages_op

def train(total_loss,global_step):
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / BATCH_SIZE
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                    global_step=global_step,
                                    decay_steps=decay_steps,
                                    decay_rate=LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    tf.summary.scalar('learning_rate',lr)

    loss_averages_op = _add_loss_summaries(total_loss)

    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)

    apply_gradient_op = opt.apply_gradients(grads,global_step)

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name,var)

    for grad,var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients',grad)

    variables_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    variables_averages_op = variables_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op,variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op