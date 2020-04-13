import tensorflow as tf 
import numpy as np
import os
from pathlib import Path

import nibabel as nib
import utils

def interpolation_test(model_file, save_dir, latent_size, n_interpolations, dimensionality):

    print('Interpolation Test ..')

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    latents_1 = np.random.normal(size=(1, latent_size))
    latents_2 = np.random.normal(size=(1, latent_size))

    latents_1 = np.tile(latents_1, (n_interpolations, 1))
    latents_2 = np.tile(latents_2, (n_interpolations, 1))

    interp_weights = np.linspace(0, 1, n_interpolations)
    interp_weights = np.expand_dims(interp_weights, axis=-1)

    interp_latents = interp_weights * latents_1 + (1 - interp_weights) * latents_2

    generator = tf.keras.models.load_model(model_file, custom_objects={'leaky_relu': tf.nn.leaky_relu})

    # for latent in interp_latents:
    generated = generator.predict([interp_latents, np.array([1.0])])

    for e, fakes in enumerate(generated):
        utils.save_generated_mri(fakes[e], str(save_dir.joinpath('{}.nii.gz'.format(e))))

def nearest_neighbor_test(model_file, tf_records_file, save_dir, latent_size, resolution, dimensionality):

    print('Nearest Neighbor Test ..')

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    latents = np.random.normal(size=(1, 1024))

    generator = tf.keras.models.load_model(model_file, custom_objects={'leaky_relu': tf.nn.leaky_relu})

    generated = generator.predict([latents, np.array([1.0])])
    utils.save_generated_mri(generated[0], str(save_dir.joinpath('generated.nii.gz')))

    train_dataset = utils.get_dataset(tf_records_file, resolution, batch_size=1, dimensionality=3)

    similarity_metric_mae = lambda x,y : tf.reduce_mean(tf.abs(x-y))

    similarity_metric_mse = lambda x,y : tf.reduce_mean(tf.square(x-y))

    print('Finding most similar ...')
    closest_img_mae = []
    closest_img_metric_mae = tf.float32.max

    closest_img_mse = []
    closest_img_metric_mse = tf.float32.max
    count = 0
    for img in train_dataset:
        count+=1
        print(count)
        if similarity_metric_mae(generated, img) < closest_img_metric_mae:
            closest_img_mae = img 
            closest_img_metric_mae = similarity_metric_mae(generated, img)

        if similarity_metric_mse(generated, img) < closest_img_metric_mse:
            closest_img_mse = img 
            closest_img_metric_mse = similarity_metric_mse(generated, img)

    print('Most similar image has MAE of {}'.format(closest_img_metric_mae))
    utils.save_generated_mri(closest_img_mae[0], str(save_dir.joinpath('closest_mae.nii.gz')), dynamic_range=[-1.0, 1.0])

    print('Most similar image has MSE of {}'.format(closest_img_metric_mse))
    utils.save_generated_mri(closest_img_mse[0], str(save_dir.joinpath('closest_mse.nii.gz')), dynamic_range=[-1.0, 1.0])

def do_test(config):
    if config.test_name == 'interpolation':
        interpolation_test(
            model_file=config.model_file, 
            save_dir=config.save_dir, 
            dimensionality=config.dimensionality, 
            latent_size=config.latent_size,
            n_interpolations=config.n_interpolations)
    elif config.test_name == 'nearest_neighbor':
        nearest_neighbor_test(
            model_file=config.model_file,  
            save_dir=config.save_dir, 
            tf_records_file=config.tf_records_file, 
            dimensionality=config.dimensionality, 
            resolution=config.resolution, 
            latent_size=config.latent_size)
    else:
        raise NotImplementedError


