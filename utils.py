import os

import tensorflow as tf

def wasserstein_loss(y_true, y_pred):
    return tf.reduce_mean(y_true * y_pred)

# def gradient_penalty_loss(y_true, y_pred, averaged_samples, gradient_penalty_weight):
# 	with tf.GradientTape() as tape:
# 		gradients = tape.gradients(y_pred, averaged_samples)
# 		gradients_squared = tf.square(gradients)

def load_img(filepath, res=2, num_channels=3):
	img = tf.io.read_file(filepath)
	img = tf.image.decode_jpeg(img, channels=num_channels)
	img = tf.image.resize(img, (2**res, 2**res))
	return img

def get_dataset(dataset_dir, res, batch_size, num_channels=3, img_ext='jpg'):

	dataset = tf.data.Dataset.list_files(os.path.join(dataset_dir, '*.'+img_ext))
	dataset = dataset.map(lambda x: load_img(x, res=res, num_channels=num_channels), num_parallel_calls=tf.data.experimental.AUTOTUNE)
	dataset = dataset.shuffle(1000)
	dataset = dataset.batch(batch_size)
	dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

	return dataset