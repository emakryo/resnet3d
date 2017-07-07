# Convert PET/CT data into TFRecord formats

import os
import numpy as np
import tensorflow as tf
import loader

FLAGS = tf.app.flags.FLAGS
#tf.app.flags.DEFINE_integer('bsize', 16, 'Length of each input')
#tf.app.flags.DEFINE_string('filename', 'data.tfrecord', 'File name for converted data')

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert(pet_data, ct_data, mask_data, writer, size=16):
    xd, yd, zd = pet_data.shape
    lung_ids = [1, 2]
    hs = int(size/2)

    count = 0
    for x,y,z in [(x,y,z)
                  for x in range(hs, xd-hs)
                  for y in range(hs, yd-hs)
                  for z in range(hs, zd-hs)
                  if mask_data[x, y, z] in lung_ids]:
        count += 1
        volume = ct_data[x-hs:x+hs, y-hs:y+hs, z-hs:z+hs]
        target = pet_data[x, y, z]
        example = tf.train.Example(features=tf.train.Features(
            feature={
                'size': _int64_feature(size),
                'volume': _bytes_feature(volume.astype(np.int16).tobytes()),
                'target': _int64_feature(target)}))
        writer.write(example.SerializeToString())

    print(count, "data")

def _main(_):
    idx = range(1,2)

    with tf.python_io.TFRecordWriter(filename) as writer:
        zipped = zip(loader.raw_PET('N', idx),
                     loader.raw_CT('N', idx),
                     loader.raw_lung_lesion_mask('N', idx))
        for pet_data, ct_data, mask_data in zipped:
            convert(pet_data, ct_data, mask_data, writer, FLAGS.block_size)

def test(_):
    filename = 'tmp.tfrecord'
    pet_data = (np.random.rand(100, 100, 100)*100).astype(int)
    ct_data = (np.random.rand(100, 100, 100)*100).astype(int)
    mask_data = (np.random.rand(100, 100, 100)*3).astype(int)
    with tf.python_io.TFRecordWriter(filename) as writer:
        convert(pet_data, ct_data, mask_data, writer, 16)

def main(_):
    normal_index = list(range(1, 11))
    save_dir = 'data/processed'
    filename = 'N%05d_N%05d_lung.tfrecord'%(normal_index[0], normal_index[-1])
    pet_data = loader.raw_PET('N', normal_index)
    ct_data = loader.raw_CT('N', normal_index)
    mask_data = loader.raw_lung_mask('N', normal_index)

    os.makedirs(save_dir, exist_ok=True)

    with tf.python_io.TFRecordWriter(os.path.join(save_dir, filename)) as writer:
        for pet, ct, mask in zip(pet_data, ct_data, mask_data):
            convert(pet, ct, mask, writer, 16)


if __name__=="__main__":
    tf.app.run(main=main)
