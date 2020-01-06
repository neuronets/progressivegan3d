import os
import tensorflow as tf

def adjust_dynamic_range(data, drange_in, drange_out):
    if drange_in != drange_out:
        scale = (drange_out[1] - drange_out[0]) / (drange_in[1] - drange_in[0])
        bias = drange_out[0] - drange_in[0] * scale
        data = data * scale + bias
    return data

def random_weight_sample(reals, fakes):
    weight_shape = (tf.shape(reals)[0],) + (1,1,1)
    weight = tf.random.uniform(weight_shape,minval=0, maxval=1)
    return (weight * reals) + ((1 - weight) * fakes)

def load_img(filepath, res=2, num_channels=3):
    img = tf.io.read_file(filepath)
    img = tf.image.decode_jpeg(img, channels=num_channels)
    img = tf.image.resize(img, (2**res, 2**res))
    img = adjust_dynamic_range(img, [0.0, 255.0], [-1.0, 1.0])
    return img

def get_dataset(dataset_dir, res, batch_size, num_channels=3, img_ext='jpg'):
    with tf.device('cpu:0'):
        dataset = tf.data.Dataset.list_files(os.path.join(dataset_dir, '*.'+img_ext))
        dataset = dataset.map(lambda x: load_img(x, res=res, num_channels=num_channels), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.shuffle(1000)
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return dataset