from trainer import PGGAN

latents_size = 256
n_classes = 1
dataset_dir = 'data/ixi_slices'

pggan = PGGAN(latents_size, n_classes, dataset_dir, num_channels=1)

pggan.train()