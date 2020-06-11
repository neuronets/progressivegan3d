import os
import numpy as np
import tensorflow as tf
import nibabel as nib
from pathlib import Path
import scipy

def adjust_dynamic_range(data, drange_in, drange_out):
    if drange_in != drange_out:
        scale = (drange_out[1] - drange_out[0]) / (drange_in[1] - drange_in[0])
        bias = drange_out[0] - drange_in[0] * scale
        data = data * scale + bias
    return data

def random_weight_sample(reals, fakes, dimensionality):
    weight_shape = (tf.shape(reals)[0],) + (1,)+(1,)*dimensionality
    weight = tf.random.uniform(weight_shape, minval=0, maxval=1)
    return (weight * reals) + ((1 - weight) * fakes)

def parse_2d_image(record, target_res):
    image_feature_description = {
        'img': tf.io.FixedLenFeature([], tf.string),
        'shape': tf.io.FixedLenFeature([3], tf.int64)
    }
    data = tf.io.parse_single_example(record, image_feature_description)
    img = data['img']
    img = tf.io.decode_raw(img, tf.uint8)
    img = tf.cast(img, tf.float32)
    img = tf.reshape(img, data['shape'])
    img = tf.image.resize(img, (2**target_res, 2**target_res))
    img = adjust_dynamic_range(img, [0.0, 255.0], [-1.0, 1.0])
    return img

def parse_3d_image(record, target_res, labels_exist=False):
    image_feature_description = {
        'img': tf.io.FixedLenFeature([], tf.string),
        'shape': tf.io.FixedLenFeature([4], tf.int64)
        
    }
    if labels_exist:
        image_feature_description['label'] = tf.io.FixedLenFeature([], tf.int64)
    data = tf.io.parse_single_example(record, image_feature_description)
    img = data['img']
    img = tf.io.decode_raw(img, tf.uint8)
    img = tf.cast(img, tf.float32)
    # img = tf.reshape(img, data['shape'])
    img = tf.reshape(img, (2**target_res, 2**target_res, 2**target_res, data['shape'][-1]))

    shape = tf.cast(data['shape'], tf.float32)

    full_res = tf.cast(tf.math.log(shape[0])/tf.math.log(2.0), tf.int32)

    img = adjust_dynamic_range(img, [0.0, 255.0], [-1.0, 1.0])

    if labels_exist:
        label = data['label']
    else:
        label = -1

    return img, label

def parse_image(record, target_res, dimensionality, labels_exist):

    if dimensionality==2:
        return parse_2d_image(record, target_res, labels_exist=labels_exist)
    else:
        return parse_3d_image(record, target_res, labels_exist=labels_exist)

def get_dataset(tf_record_dir, res, batch_size, dimensionality, labels_exist=False):
    with tf.device('cpu:0'):
        dataset = tf.data.Dataset.list_files(os.path.join(tf_record_dir, 'resolution-%03d-*.tfrecord'%(2**(res))))

        dataset = dataset.shuffle(20)
        dataset = dataset.interleave(lambda file: tf.data.TFRecordDataset(file, compression_type='GZIP'),
                         cycle_length=tf.data.experimental.AUTOTUNE, block_length=4)
        dataset = dataset.map(lambda x: parse_image(x, target_res=res, dimensionality=dimensionality, labels_exist=labels_exist), 
                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.shuffle(100)
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return dataset

def save_generated_mri(generated, filename, dynamic_range=[-1, 1]):
    generated = adjust_dynamic_range(generated, dynamic_range, [0.0, 255.0])
    generated = tf.clip_by_value(generated, 0.0, 255.0)
    img_arr = np.squeeze(np.array(generated)).astype(np.uint8)
    mri = nib.Nifti1Image(img_arr, np.eye(4))
    nib.save(mri, filename)

def generate(config):
    run_id = Path(config.run_id)
    model_dir = run_id.joinpath(config.model_dir)
    generated_dir = run_id.joinpath(config.generated_dir)
    start_resolution_log = int(np.log2(config.start_resolution))
    target_resolution_log = int(np.log2(config.target_resolution))

    generated_dir.mkdir(exist_ok=True)

    for res in range(start_resolution_log+1, target_resolution_log+1):
        generator = tf.keras.models.load_model(str(model_dir.joinpath('g_{}.h5'.format(res))), custom_objects={'leaky_relu': tf.nn.leaky_relu})
        for i in range(config.num_samples):
            latents = tf.random.normal((1, config.latent_size))
            fakes = generator([latents, 1.0])
            save_generated_mri(fakes[0], str(generated_dir.joinpath('res_{}_{}.nii.gz'.format(res, i))))

