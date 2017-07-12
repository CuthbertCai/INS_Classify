# -*- coding: utf-8 -*-

# This module evaluate the CNN with the checkpoint.

# You can choose the deep CNN with batch normalization,
# simple CNN with batch normalization or simple CNN without
# batch normalization for train
from datetime import datetime
import tensorflow as tf
import numpy as np
import math
import time
import ins_small_model_bn
import ins_image_input

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_images_dir','/home/cuthbert/Program/INS_Classify/image/test',
                           """Directory where to get the images in JPG format""")
tf.app.flags.DEFINE_string('eval_dir', '/home/cuthbert/Program/INS_Classify/eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test.tfrecords',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/home/cuthbert/Program/INS_Classify/train',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 13826,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                         """Whether to run eval only once.""")

def eval_once(saver,summary_writer,top_k_op,summary_op):
    global threads
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess,ckpt.model_checkpoint_path)
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('No checkpoint file found')
            return

        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess,coord=coord,daemon=True,start=True))

            num_iter = int(math.ceil(FLAGS.num_examples / ins_small_model_bn.BATCH_SIZE))
            true_count = 0
            total_sample_count = num_iter * ins_small_model_bn.BATCH_SIZE
            step = 0
            while step < num_iter and not coord.should_stop():
                predicitons = sess.run([top_k_op])
                true_count += np.sum(predicitons)
                step += 1

            precision = true_count / total_sample_count
            print('%s: precision @ 1 = %.3f'  %(datetime.now(),precision))

            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary.value.add(tag='Precision @ 1',simple_value = precision)
            summary_writer.add_summary(summary,global_step)
        except Exception as e:
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads,stop_grace_period_secs=10)

def evaluate():
    with tf.Graph().as_default() as g:
        images,labels = ins_image_input.eval_inputs(FLAGS.eval_data,ins_small_model_bn.BATCH_SIZE)
        logits = ins_small_model_bn.inference(images,istrain=False)
        top_k_op = tf.nn.in_top_k(logits,labels,1)
        variable_averages = tf.train.ExponentialMovingAverage(ins_small_model_bn.MOVING_AVERAGE_DECAY)
        variables_to_store = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_store)

        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(FLAGS.eval_dir,g)

        while True:
            eval_once(saver,summary_writer,top_k_op,summary_op)
            if FLAGS.run_once:
                break
            time.sleep(FLAGS.eval_interval_secs)

def main(argv=None):
    ins_image_input.convert_to_records(FLAGS.eval_images_dir,'test')
    if tf.gfile.Exists(FLAGS.eval_dir):
        tf.gfile.DeleteRecursively(FLAGS.eval_dir)
    tf.gfile.MakeDirs(FLAGS.eval_dir)
    evaluate()


if __name__ == '__main__':
    tf.app.run()
