# Convert PET/CT data into TFRecord formats

import numpy as np
import tensorflow as tf
import loader

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('bsize', 16, 'Length of each input')
tf.app.flags.DEFINE_string('filename', 'data.tfrecord', 'File name for converted data')

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert(pet_data, ct_data, mask_data, writer, size=16):
    xd, yd, zd = pet_data.shape
    lung_id = 2
    hs = int(size/2)

    for x,y,z in [(x,y,z)
                  for x in range(hs, xd-hs)
                  for y in range(hs, yd-hs)
                  for z in range(hs, zd-hs)
                  if mask_data[x, y, z] == lung_id]:
        volume = ct_data[x-hs:x+hs, y-hs:y+hs, z-hs:z+hs]
        target = pet_data[x, y, z]
        example = tf.train.Example(features=tf.train.Features(
            feature={
                'size': _int64_feature(size),
                'volume': _bytes_feature(volume.astype(np.int32).tobytes()),
                'target': _int64_feature(target)}))
        writer.write(example.SerializeToString())

def _main(_):
    idx = range(1,2)

    with tf.python_io.TFRecordWriter(filename) as writer:
        zipped = zip(loader.raw_PET('N', idx),
                     loader.raw_CT('N', idx),
                     loader.raw_lung_lesion_mask('N', idx))
        for pet_data, ct_data, mask_data in zipped:
            convert(pet_data, ct_data, mask_data, writer, FLAGS.block_size)

def main(_):
    filename = 'tmp.tfrecord'
    pet_data = (np.random.rand(100, 100, 100)*100).astype(int)
    ct_data = (np.random.rand(100, 100, 100)*100).astype(int)
    mask_data = (np.random.rand(100, 100, 100)*3).astype(int)
    with tf.python_io.TFRecordWriter(filename) as writer:
        convert(pet_data, ct_data, mask_data, writer, 16)

if __name__=="__main__":
    tf.app.run(main=main)
