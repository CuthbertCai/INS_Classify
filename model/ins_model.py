# -*- coding" utf-8 -*-

# This module is a deep CNN with 5 convolution layer,
# 2 full connected layer and 1 output layer.
# Batch nornalization is used in the deep CNN.

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

def _variable_with_weight_decay(name,shape,wd):
    var = tf.get_variable(name,shape,initializer=tf.contrib.layers.xavier_initializer())
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var),wd,name='weight_loss')
        tf.add_to_collection('loss',weight_decay)
    return var

def batch_norm(x, out_dim, is_train):
    with tf.variable_scope('batch_norm') as scope:
        gamma = tf.get_variable('gamma', shape=out_dim, initializer=tf.constant_initializer(1.0))
        beta = tf.get_variable('beta', shape=out_dim, initializer=tf.constant_initializer(0.0))
        axes = list(range(len(x.get_shape()) -1))
        batch_mean,batch_variance = tf.nn.moments(x,axes=axes,name='moments')
        ema = tf.train.ExponentialMovingAverage(0.5)
        ema_apply_op = ema.apply([batch_mean, batch_variance])

        def update():
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean),tf.identity(batch_variance)

        if is_train:
            mean, variance = update()
        else:
            mean = ema.average(batch_mean)
            variance = ema.average(batch_variance)
        bn = tf.nn.batch_normalization(x,mean,variance,beta,gamma,0.001)
    return bn


def inference(images, keep_pro = 1, is_train=True):
    with tf.variable_scope('conv1') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[5, 5, 3, 64],
                                             wd=0.0)
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', [64], initializer=tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, biases)
        bn = batch_norm(pre_activation,[64], is_train=is_train)
        conv1 = tf.nn.relu(bn, name=scope.name)
        _activation_summary(conv1)

        # pool1
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool1')

    # conv2
    with tf.variable_scope('conv2') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[5, 5, 64, 64],
                                             wd=0.0)
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', [64], initializer=tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, biases)
        bn = batch_norm(pre_activation,[64],is_train=is_train)
        conv2 = tf.nn.relu(bn, name=scope.name)
        _activation_summary(conv2)

    # pool2
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    with tf.variable_scope('conv3') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[3,3,64,64],
                                             wd=0.0)
        conv = tf.nn.conv2d(pool2,kernel,[1,1,1,1],padding='SAME')
        bias = tf.get_variable('bias',[64],initializer=tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv,bias)
        bn = batch_norm(pre_activation, [64], is_train)
        conv3 = tf.nn.relu(bn,name=scope.name)
        _activation_summary(conv3)

    #pool3 = tf.nn.max_pool(conv3, ksize=[1,3,3,1],
    #                      strides=[1,2,2,1], padding='SAME', name='pool3')

    with tf.variable_scope('conv4') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[3,3,64,192],
                                             wd=0.0)
        conv = tf.nn.conv2d(conv3,kernel,[1,1,1,1],padding='SAME')
        bias = tf.get_variable('bias',[192],initializer=tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv,bias)
        bn = batch_norm(pre_activation, [192], is_train)
        conv4 = tf.nn.relu(bn,name=scope.name)
        _activation_summary(conv4)

    with tf.variable_scope('conv5') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[3,3,192,256],
                                             wd=0.0)
        conv = tf.nn.conv2d(conv4,kernel,[1,1,1,1],padding='SAME')
        bias = tf.get_variable('bias',[256],initializer=tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv,bias=bias)
        bn = batch_norm(pre_activation, [256], is_train)
        conv5 = tf.nn.relu(bn,name=scope.name)
        _activation_summary(conv5)

    pool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1], padding='SAME', name='pool5')

    # local6
    with tf.variable_scope('local6') as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        reshape = tf.reshape(pool5, [BATCH_SIZE, -1])
        dim = reshape.get_shape()[1].value
        weights = _variable_with_weight_decay('weights', shape=[dim, 1024], wd=0.002)
        biases = tf.get_variable('biases', [1024], initializer=tf.constant_initializer(0.1))
        pre_activation = tf.matmul(reshape, weights) + biases
        bn = batch_norm(pre_activation,[1024],is_train)
        local6 = tf.nn.relu(bn, name=scope.name)
        _activation_summary(local6)

    with tf.variable_scope('dropout') as scope:
        dropout = tf.nn.dropout(local6,keep_prob=keep_pro)

    # local7
    with tf.variable_scope('local7') as scope:
        weights = _variable_with_weight_decay('weights', shape=[1024, 256], wd=0.002)
        biases = tf.get_variable('biases', [256], initializer=tf.constant_initializer(0.1))
        pre_activation = tf.matmul(dropout, weights) + biases
        bn = batch_norm(pre_activation, [256], is_train)
        local7 = tf.nn.relu(bn, name=scope.name)
        _activation_summary(local7)

    # linear layer(WX + b),
    # We don't apply softmax here because
    # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
    # and performs the softmax internally for efficiency.
    with tf.variable_scope('softmax_linear') as scope:
        weights = _variable_with_weight_decay('weights', [256, CLASSES_NUM], wd=0.0)
        biases = tf.get_variable('biases', [CLASSES_NUM],initializer=
                                  tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(local7, weights), biases, name=scope.name)
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