# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Evaluation for CIFAR-10.

Accuracy:
cifar10_train.py achieves 83.0% accuracy after 100K steps (256 epochs
of data) as judged by cifar10_eval.py.

Speed:
On a single Tesla K40, cifar10_train.py processes a single batch of 128 images
in 0.25-0.35 sec (i.e. 350 - 600 images /sec). The model reaches ~86%
accuracy after 100K steps in 8 hours of training time.

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf

import cifar10
import argparse

parser = argparse.ArgumentParser()


def str2bool(string):
  if string == 'False':
    return False
  elif string == 'True':
    return True


parser = argparse.ArgumentParser()

parser.add_argument("--eval_dir", type=str,
                    help="Directory where to write event logs.",
                    default="/tmp/cifar10_eval")
parser.add_argument("--eval_data", type=str,
                    help="Either 'test' or 'train_eval'.",
                    default="test")
parser.add_argument("--checkpoint_dir", type=str,
                    help="Directory where to read model checkpoints.",
                    default="/tmp/cifar10_train")
parser.add_argument("--eval_interval_secs", type=int,
                    help="How often to run the eval.",
                    default=60 * 5)
parser.add_argument("--num_examples", type=int,
                    help="Number of examples to run.",
                    default=10000)
parser.add_argument("--run_once", type=str2bool,
                    help="Whether to run eval only once.",
                    default=False, choices=[True, False])

# Basic model parameters (used in cifar10.py).
parser.add_argument("--batch_size", type=int,
                    help="Number of images to process in a batch.",
                    default=128)
parser.add_argument("--data_dir", type=str,
                    help="Path to the CIFAR-10 data directory.",
                    default='/tmp/cifar10_data')
parser.add_argument("--use_fp16", type=str2bool,
                    help="Train the model using fp16.",
                    default=False, choices=[True, False])
parser.add_argument("--use_linear_model", type=str2bool,
                    help="Whether to use a simple linear model.",
                    default=False, choices=[True, False])
parser.add_argument("--use_greyscale", type=str2bool,
                    help="Whether to transform the images to greyscale (only for linear model).",
                    default=False, choices=[True, False])
parser.add_argument("--Wwd", type=float,
                    help="Weight decay of the linear model.",
                    default=0.1)
parser.add_argument("--use_suquan", type=str2bool,
                    help="Whether to perform a quantile normalization (only for linear model).",
                    default=False, choices=[True, False])
parser.add_argument("--optimize_f", type=str2bool,
                    help="Whether to optimize the quantile function (only for linear model).",
                    default=False, choices=[True, False])
parser.add_argument("--Fwd", type=float,
                    help="Weight decay of the quantile function.", default=0.1)
parser.add_argument("--f_init", type=str,
                    help="Initial quantile function.", default='constant',
                    choices=["normal", "uniform", "constant"])
parser.add_argument("--moving_average_decay", type=float,
                    help="The decay to use for the moving average.",
                    default=0.999)
parser.add_argument("--num_epochs_per_decay", type=float,
                    help="Epochs after which learning rate decays.",
                    default=350.0)
parser.add_argument("--learning_rate_decay_factor", type=float,
                    help="Learning rate decay factor.",
                    default=0.1)
parser.add_argument("--initial_learning_rate", type=float,
                    help="Initial learning rate.",
                    default=0.1)

args = parser.parse_args()


def eval_once(saver, summary_writer, top_k_op, summary_op, args):
  """Run Eval once.

  Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
  """
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(args.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/cifar10_train/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print('No checkpoint file found')
      return

    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

      num_iter = int(math.ceil(args.num_examples / args.batch_size))
      true_count = 0  # Counts the number of correct predictions.
      total_sample_count = num_iter * args.batch_size
      step = 0
      while step < num_iter and not coord.should_stop():
        predictions = sess.run([top_k_op])
        true_count += np.sum(predictions)
        step += 1

      # Compute precision @ 1.
      precision = true_count / total_sample_count
      print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      summary.value.add(tag='Precision @ 1', simple_value=precision)
      summary_writer.add_summary(summary, global_step)
    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)


def evaluate(args):
  """Eval CIFAR-10 for a number of steps."""
  with tf.Graph().as_default() as g:
    # Get images and labels for CIFAR-10.
    eval_data = args.eval_data == 'test'
    images, labels = cifar10.inputs(eval_data=eval_data, args=args)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = cifar10.inference(images, args)

    # Calculate predictions.
    top_k_op = tf.nn.in_top_k(logits, labels, 1)

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        args.moving_average_decay)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter(args.eval_dir, g)

    while True:
      eval_once(saver, summary_writer, top_k_op, summary_op, args)
      if args.run_once:
        break
      time.sleep(args.eval_interval_secs)


def main(argv=None):  # pylint: disable=unused-argument
  cifar10.maybe_download_and_extract(args)
  if tf.gfile.Exists(args.eval_dir):
    tf.gfile.DeleteRecursively(args.eval_dir)
  tf.gfile.MakeDirs(args.eval_dir)
  evaluate(args)


if __name__ == '__main__':
  tf.app.run()
