# -*- coding: utf-8 -*-

# This module feed images for both train model and evaluate model.

import os
import tensorflow as tf
from PIL import Image
import ins_model

# All images in our implementation are divided into 12 classes
classes = ["lion", "phone", "car", "bicycle", "sneaker", "airplane",
           "computer", "human", "dog", "watch", "flower", "camera"]
HEIGHT = 64
WIDTH = 64
DEPTH = 3

def convert_to_records(dir, name):
    # We convert all the images in JPG format into tfrecords
    # which is the standard format in tensorflow

    filename = name + '.tfrecords'
    if not os.path.exists(filename):
        writer = tf.python_io.TFRecordWriter(filename)
        for index, name in enumerate(classes):
            class_path = dir + '/' + name + '/'
            for img_name in os.listdir(class_path):
                img_path = class_path + img_name
                img = Image.open(img_path)
                img = img.resize((100, 100))
                img_raw = img.tobytes()
                example = tf.train.Example(features=tf.train.Features(feature={
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[int(index)])),
                    "img_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                }))
                writer.write(example.SerializeToString())
        writer.close()


def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                           'label': tf.FixedLenFeature([], tf.int64),
                                       })

    image = tf.decode_raw(features['img_raw'], tf.uint8)
    image = tf.reshape(image, [100,100,3])
    label = tf.cast(features['label'], tf.int32)

    return image, label

def train_inputs(filename, batch_size):
    # Images are feeded into train model with the filename_queue
    # Augmentation is used to increase train samples

    with tf.name_scope('train_input'):
        filename_queue = tf.train.string_input_producer([filename])
        image, label = read_and_decode(filename_queue)

        image = tf.cast(image, tf.float32)

        distorted_image = tf.random_crop(image,size=[HEIGHT,WIDTH,3])
        distorted_image = tf.image.random_flip_left_right(distorted_image)
        distorted_image = tf.image.random_brightness(distorted_image,max_delta=63)
        distorted_image = tf.image.random_contrast(distorted_image,lower=0.2,upper=1.8)
        float_image = tf.image.per_image_standardization(distorted_image)
        float_image.set_shape([HEIGHT,WIDTH,DEPTH])

        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(ins_model.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * min_fraction_of_examples_in_queue)

        print('Filling queue with %d images before starting to train. '
              'This will take a few minutes.' % min_queue_examples)

        images, sparse_labels = tf.train.shuffle_batch(
            [float_image, label],
            batch_size=batch_size,
            num_threads=2,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples
        )
        # display the images in visualizer
        tf.summary.image('images', images)

        return images, sparse_labels

def eval_inputs(filename, batch_size):
    # Images are feeded into evaluate model with the filename_queue
    # Augmentation is used to increase evaluate samples

    with tf.name_scope('eval_input'):
        filename_queue = tf.train.string_input_producer([filename])
        image, label = read_and_decode(filename_queue)

        image = tf.cast(image, tf.float32)
        image = tf.image.resize_image_with_crop_or_pad(image,target_height=HEIGHT,target_width=WIDTH)
        float_image = tf.image.per_image_standardization(image)

        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(ins_model.NUM_EXAMPLES_PER_EPOCH_FOR_TEST * min_fraction_of_examples_in_queue)

        images, sparse_labels = tf.train.shuffle_batch(
            [float_image, label],
            batch_size=batch_size,
            num_threads=2,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples
        )
        # display the images in visualizer
        tf.summary.image('images', images)

        return images, sparse_labels
