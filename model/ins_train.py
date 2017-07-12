# -*- coding: utf-8 -*-

# This module trains the CNN and records the checkpoint.

# You can choose the deep CNN with batch normalization,
# simple CNN with batch normalization or simple CNN without
# batch normalization for train
import ins_small_model_bn
import ins_image_input
from datetime import datetime
import tensorflow as tf
import time

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_images_dir','/home/cuthbert/Program/INS_Classify/image/train',
                           """Directory where to get the images in JPG format""")
tf.app.flags.DEFINE_string('train_dir', '/home/cuthbert/Program/INS_Classify/train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 10,
                            """How often to log results to the console.""")

def train():
    with tf.Graph().as_default():
        globel_step = tf.contrib.framework.get_or_create_global_step()
        images,labels = ins_image_input.train_inputs('train.tfrecords',ins_small_model_bn.BATCH_SIZE)
        logits = ins_small_model_bn.inference(images)
        loss = ins_small_model_bn.loss(logits,labels)
        train_op = ins_small_model_bn.train(loss,globel_step)

        class _LoggerHook(tf.train.SessionRunHook):
            """Logs loss and runtime."""

            def begin(self):
                self._step = -1
                self._start_time = time.time()

            def before_run(self, run_context):
                self._step += 1
                return tf.train.SessionRunArgs(loss)  # Asks for loss value.

            def after_run(self, run_context, run_values):
                if self._step % FLAGS.log_frequency == 0:
                    current_time = time.time()
                    duration = current_time - self._start_time
                    self._start_time = current_time

                    loss_value = run_values.results
                    examples_per_sec = FLAGS.log_frequency * ins_small_model_bn.BATCH_SIZE / duration
                    sec_per_batch = float(duration / FLAGS.log_frequency)

                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                                  'sec/batch)')
                    print(format_str % (datetime.now(), self._step, loss_value,
                                        examples_per_sec, sec_per_batch))

        with tf.train.MonitoredTrainingSession(
                checkpoint_dir=FLAGS.train_dir,
                hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
                       tf.train.NanTensorHook(loss),
                       _LoggerHook()],
                config=tf.ConfigProto(
                    log_device_placement=FLAGS.log_device_placement)) as mon_sess:
            while not mon_sess.should_stop():
                mon_sess.run(train_op)

def main(argv=None):
    ins_image_input.convert_to_records(FLAGS.train_images_dir,'train')
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()
if __name__ == '__main__':
    tf.app.run()
