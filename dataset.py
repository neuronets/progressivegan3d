from pathlib import Path
import time

import numpy as np
import tensorflow as tf
from PIL import Image
import nibabel as nib

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def serialize_example(img, shape):
    feature = {
        'img' : _bytes_feature(img),
        'shape' : _int64_feature(shape)
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def prepare_2d_tf_record_dataset(dataset_dir, tf_record_filename):

    tf_record_writer = tf.io.TFRecordWriter(tf_record_filename)

    dataset_dir = Path(dataset_dir)
    img_filenames = dataset_dir.glob('*.jpg')

    n_images = len(img_filenames)

    for e, f in enumerate(img_filenames):
        print('{} / {} images done'.format(e, n_images))
        img = Image.open(f)

        img = np.array(img).astype(np.uint8)
        img_data = img.ravel().tostring()
        # img_data = tf.image.encode_png(img).tostring()
        img_shape = img.shape
        if len(img_shape) == 2:
            img_shape += (1,)

        tf_record_writer.write(serialize_example(img_data, img_shape))


def prepare_3d_tf_record_dataset(dataset_dir, tf_record_filename):

    tf_record_writer = tf.io.TFRecordWriter(tf_record_filename)

    dataset_dir = Path(dataset_dir)
    img_filenames = dataset_dir.glob('*orig*.nii.gz')

    n_images = len(img_filenames)

    print('{} images found'.format(n_images))
    start = time.time()
    for e, f in enumerate(img_filenames):
        print('{} / {} images done'.format(e, n_images))

        img = nib.load(f)
        # img_data = img.get_fdata().astype(np.uint8)
        img_data = img.get_fdata()
        img_data = (255 * (img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data)) ).astype(np.uint8)
        img_shape = np.array(img_data.shape).astype(np.int64)
        img_data = img_data.ravel().tostring()
        tf_record_writer.write(serialize_example(img_data, img_shape))
    print('Time taken : {}'.format(time.time()-start))


def prepare_tf_record_dataset(dataset_dir, save_path, dimensionality):
    if dimensionality==2:
        return prepare_2d_tf_record_dataset(dataset_dir, save_path)
    else:
        return prepare_3d_tf_record_dataset(dataset_dir, save_path)

