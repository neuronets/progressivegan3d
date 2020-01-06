from trainer import PGGAN
import tensorflow as tf

latents_size = 256
n_classes = 1
dataset_dir = 'data/ixi_slices'
gpu = '/gpu:1'

with tf.device(gpu):
    pggan = PGGAN(
        latents_size=latents_size, 
        dataset_dir=dataset_dir, 
        n_classes=1, 
        num_channels=1, 
        run_id='1',
        target_resolution=256)

    pggan.train()