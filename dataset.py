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

def serialize_example(img, shape, label):
    feature = {
        'img' : _bytes_feature(img),
        'shape' : _int64_feature(shape),
        'label' : _int64_feature([label])
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def prepare_2d_tf_record_dataset(dataset_dir, tf_record_filename, glob_ext, n_img_per_shard):

    dataset_dir = Path(dataset_dir)
    img_filenames = list(dataset_dir.glob(glob_ext))

    tf_record_save_dir = Path(tf_record_save_dir)

    n_images = len(img_filenames)
    n_shards = n_images/n_img_per_shard+1

    print('{} images found'.format(n_images))
    print('{} shards will be created'.format(n_shards))
    print('Storing {} images in each shard'.format(n_img_per_shard))

    start = time.time()

    start = time.time()
    img_count = 0

    start_resolution_log = int(np.log2(start_resolution))
    target_resolution_log = int(np.log2(target_resolution))
    resolutions = [2**res for res in range(start_resolution_log, target_resolution_log+1)]

    tf_record_filenames = []

    for shard in range(n_shards):
        print('Creating {} / {} shard '.format(shard+1, n_shards))
        
        tf_record_writers = {}
        for res in resolutions:
            tf_record_filename =  str(tf_record_save_dir.joinpath('resolution-%03d-data-%03d-of-%03d.tfrecord'%(res,
                shard+1, n_shards)))

            tf_record_writers[res] = tf.io.TFRecordWriter(tf_record_filename, options=tf_record_writer_options)

        for e, f in enumerate(img_filenames[img_count:img_count+n_img_per_shard]):
            img = Image.open(f)

            img = np.array(img)

            img_data = (255 * (img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data)))
            img_data_raveled = img_data.astype(np.uint8).ravel().tostring()
            img_shape = img.shape
            if len(img_shape)==3:
                img_shape = np.append(img_shape, 1)

            tf_record_writers[target_resolution].write(serialize_example(img_data_raveled, img_shape, img_label))

            for res in range(2, target_resolution_log):
                img_data = img_data[0::2,0::2]+img_data[0::2,1::2]+img_data[1::2,0::2]+img_data[1::2,1::2]
                img_data = (img_data) * 0.25
                img_data = (255 * (img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data)))

                actual_resolution_log = target_resolution_log-res+1
                actual_resolution = 2**actual_resolution_log

                img_data_raveled = img_data.astype(np.uint8).ravel().tostring()

                tf_record_writers[actual_resolution].write(serialize_example(img_data_raveled, img_shape, img_label))

                img_count+=1
                print('{} / {} images done'.format(img_count, n_images))
    
        [tf_record_writers[writer].close() for writer in tf_record_writers]
        print('Time taken for shard: {}'.format(time.time()-start))
    print('Total time taken: {}'.format(time.time()-start))


def prepare_3d_tf_record_dataset(dataset_dir, tf_record_save_dir, glob_ext, n_img_per_shard, 
    start_resolution=4, target_resolution=256):

    dataset_dir = Path(dataset_dir)
    img_filenames = list(dataset_dir.glob(glob_ext))

    tf_record_save_dir = Path(tf_record_save_dir)
    tf_record_save_dir.mkdir(parents=True, exist_ok=False)

    tf_record_writer_options = tf.io.TFRecordOptions(compression_type='GZIP')

    n_images = len(img_filenames)
    n_shards = n_images//n_img_per_shard+1

    print('{} images found'.format(n_images))
    print('{} shards will be created'.format(n_shards))
    print('Storing {} images in each shard'.format(n_img_per_shard))

    if n_images == 0:
        print('No images found!')
        return

    start = time.time()
    img_count = 0

    start_resolution_log = int(np.log2(start_resolution))
    target_resolution_log = int(np.log2(target_resolution))
    resolutions = [2**res for res in range(start_resolution_log, target_resolution_log+1)]

    tf_record_filenames = []

    for shard in range(n_shards):
        print('Creating {} / {} shard '.format(shard+1, n_shards))
        
        tf_record_writers = {}
        for res in resolutions:
            tf_record_filename =  str(tf_record_save_dir.joinpath('resolution-%03d-data-%03d-of-%03d.tfrecord'%(res,
                shard+1, n_shards)))

            tf_record_writers[res] = tf.io.TFRecordWriter(tf_record_filename, options=tf_record_writer_options)

        for e, f in enumerate(img_filenames[img_count:img_count+n_img_per_shard]):
            img = nib.load(str(f))
            img = nib.as_closest_canonical(img)
            img_data = img.get_fdata()
            img_data = (255 * (img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data)))
            img_shape = np.array(img_data.shape).astype(np.int64)

            if len(img_shape) == 3:
                img_data = np.expand_dims(img_data, axis=-1)

            if len(img_shape)==3:
                img_shape = np.append(img_shape, 1)

            img_label = -1
            
            ## Write img label code here if used
            # if 'T1' in str(f):
            #     img_label=0
            # elif 'T2' in str(f):
            #     img_label=1
            # else:
            #     raise ValueError('Label does not exist')

            img_data_raveled = img_data.astype(np.uint8).ravel().tostring()
            tf_record_writers[target_resolution].write(serialize_example(img_data_raveled, img_shape, img_label))

            for res in range(2, target_resolution_log):
                img_data = img_data[0::2,0::2,0::2,:]+img_data[0::2,0::2,1::2,:]+img_data[0::2,1::2,0::2,:]+img_data[0::2,1::2,1::2,:] \
                                +img_data[1::2,0::2,0::2,:]+img_data[1::2,0::2,1::2,:]+img_data[1::2,1::2,0::2,:]+img_data[1::2,1::2,1::2,:]
                img_data = (img_data) * 0.125
                img_data = (255 * (img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data)))

                actual_resolution_log = target_resolution_log-res+1
                actual_resolution = 2**actual_resolution_log

                img_data_raveled = img_data.astype(np.uint8).ravel().tostring()

                tf_record_writers[actual_resolution].write(serialize_example(img_data_raveled, img_shape, img_label))

            img_count+=1
            print('{} / {} images done'.format(img_count, n_images))

        [tf_record_writers[writer].close() for writer in tf_record_writers]
        print('Time taken for shard: {}'.format(time.time()-start))

    print('Total time taken: {}'.format(time.time()-start))

def prepare_tf_record_dataset(dataset_dir, tf_record_save_dir, dimensionality, glob_ext, n_img_per_shard, start_resolution, target_resolution):
    if dimensionality==2:
        return prepare_2d_tf_record_dataset(dataset_dir, tf_record_save_dir, glob_ext, n_img_per_shard, start_resolution, target_resolution)
    else:
        return prepare_3d_tf_record_dataset(dataset_dir, tf_record_save_dir, glob_ext, n_img_per_shard, start_resolution, target_resolution)

