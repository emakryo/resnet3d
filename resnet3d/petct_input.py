# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
# 
# Modified for 3D PET-CT data from https://github.com/tensorflow/models/tree/master/resnet

"""PET/CT dataset input module."""

import tensorflow as tf

def build_input(data_path, batch_size, size, mode):
  """Build image and targets.

  Args:
    data_path: Filename for data.
    batch_size: Input batch size.
    mode: Either 'train' or 'eval'.
  Returns:
    volumes: Batches of volumes. [batch_size, volume_size, volume_size, volume_size]
    targets: Batches of targets. [batch_size]
  Raises:
    ValueError: when the specified dataset is not supported.
  """

  data_files = tf.gfile.Glob(data_path)
  file_queue = tf.train.string_input_producer(data_files, num_epochs=50, shuffle=True)
  # Read examples from files in the filename queue.
  reader = tf.TFRecordReader()
  _, value = reader.read(file_queue)
  features = tf.parse_single_example(
      value,
      features={'volume': tf.FixedLenFeature([], tf.string),
                'target': tf.FixedLenFeature([], tf.int64)})
  volume = tf.cast(tf.reshape(tf.decode_raw(features['volume'], tf.int16),
                              (size, size, size)),
                   tf.float32)
  target = tf.cast(features['target'], tf.float32)

  if mode == 'train':
    example_queue = tf.RandomShuffleQueue(
        capacity=16 * batch_size,
        min_after_dequeue=8 * batch_size,
        dtypes=[tf.float32, tf.float32],
        shapes=[[size, size, size], []])
    num_threads = 16
  else:
    example_queue = tf.FIFOQueue(
        3 * batch_size,
        dtypes=[tf.float32, tf.float32],
        shapes=[[size, size, size], []])
    num_threads = 1

  example_enqueue_op = example_queue.enqueue([volume, target])
  tf.train.add_queue_runner(tf.train.queue_runner.QueueRunner(
      example_queue, [example_enqueue_op] * num_threads))

  # Read 'batch' labels + images from the example queue.
  volumes, targets = example_queue.dequeue_many(batch_size)

  assert len(volumes.get_shape()) == 4
  assert volumes.get_shape()[0] == batch_size
  assert len(targets.get_shape()) == 1
  assert targets.get_shape()[0] == batch_size

  # Display the training images in the visualizer.
  # tf.summary.image('images', images)
  return volumes, targets

def test():
  """debugging code"""
  vol, tgt = build_input('./tmp.tfrecord', 4, 16, 'train')
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    vol = sess.run(vol)
    print(vol[0,0,0,0], vol.shape)
