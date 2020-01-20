import os
import numpy as np
import tensorflow as tf
import nibabel as nib
import scipy

def adjust_dynamic_range(data, drange_in, drange_out):
    if drange_in != drange_out:
        scale = (drange_out[1] - drange_out[0]) / (drange_in[1] - drange_in[0])
        bias = drange_out[0] - drange_in[0] * scale
        data = data * scale + bias
    return data

def random_weight_sample(reals, fakes):
    weight_shape = (tf.shape(reals)[0],) + (1,1,1,1)
    weight = tf.random.uniform(weight_shape, minval=0, maxval=1)
    return (weight * reals) + ((1 - weight) * fakes)

def load_img(filepath, res=2, num_channels=3):
    img = tf.io.read_file(filepath)
    img = tf.image.decode_jpeg(img, channels=num_channels)
    img = tf.image.resize(img, (2**res, 2**res))
    img = adjust_dynamic_range(img, [0.0, 255.0], [-1.0, 1.0])
    return img

# def load_3d_img(filepath, res=2, num_channels=3, full_res=8):
#     img = nib.load(np.array(tf.convert_to_tensor(filepath, dtype=tf.string))).get_data()
#     img = tf.cast(img, tf.float32)
#     img = tf.reshape(img, (2**full_res, 2**full_res, 2**full_res))
#     for i in range(full_res-res):
#         img = img[0::2,0::2,0::2]+img[0::2,0::2,1::2]+img[0::2,1::2,0::2]+img[0::2,1::2,1::2] \
#                         +img[1::2,0::2,0::2]+img[1::2,0::2,1::2]+img[1::2,1::2,0::2]+img[1::2,1::2,1::2]
#         img = (img) * 0.125
#     img = adjust_dynamic_range(img, [0.0, 255.0], [-1.0, 1.0])
#     return img

def parse_3d_image(example_proto):
    image_feature_description = {
    'img': tf.io.FixedLenFeature([], tf.string),
    'shape': tf.io.FixedLenFeature([3], tf.int64)
    }
    data = tf.io.parse_single_example(example_proto, image_feature_description)
    mri = data['img']
    mri = tf.io.decode_raw(mri, tf.uint8)
    mri = tf.cast(mri, tf.float32)
    mri = tf.reshape(mri, data['shape'])
    return mri

def resize_3d_image(img, full_res, target_res):
    for i in range(full_res-target_res):
        img = img[0::2,0::2,0::2]+img[0::2,0::2,1::2]+img[0::2,1::2,0::2]+img[0::2,1::2,1::2] \
                        +img[1::2,0::2,0::2]+img[1::2,0::2,1::2]+img[1::2,1::2,0::2]+img[1::2,1::2,1::2]
        img = (img) * 0.125
    img = adjust_dynamic_range(img, [0.0, 255.0], [-1.0, 1.0])
    img = tf.expand_dims(img, axis=-1)
    return img

def get_dataset(dataset_dir, res, batch_size, num_channels=3, dimensionality=3, img_ext='jpg'):
    with tf.device('cpu:0'):
        dataset = tf.data.Dataset.list_files(os.path.join(dataset_dir, '*.'+img_ext))
        dataset = dataset.map(lambda x: load_img(x, res=res, num_channels=num_channels), 
            num_parallel_calls=tf.data.experimental.AUTOTUNE)

        dataset = dataset.shuffle(200)
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return dataset

def get_dataset_3d(dataset_dir, res, batch_size):
    with tf.device('cpu:0'):
        dataset = tf.data.TFRecordDataset(dataset_dir)
        dataset = dataset.map(parse_3d_image, 
                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(lambda x: resize_3d_image(x, full_res=8, target_res=res), 
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.shuffle(200)
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return dataset

# def get_3d_dataset(dataset_dir, res, batch_size, num_channels=3, img_ext='jpg'):
#     with tf.device('cpu:0'):
#         dataset = tf.data.Dataset.list_files(os.path.join(dataset_dir, '*.'+img_ext))
#         dataset = dataset.map(lambda x: load_img(x, res=res, num_channels=num_channels), num_parallel_calls=tf.data.experimental.AUTOTUNE)
#         dataset = dataset.shuffle(1000)
#         dataset = dataset.batch(batch_size, drop_remainder=True)
#         dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
#         return dataset

# def get_tf_record_dataset(tf_records_file, res, batch_size):
#     pass


# def prepare_mri_tf_record_dataset(dataset_dir):
#     import nibabel as nib