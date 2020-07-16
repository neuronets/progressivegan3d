import os
import tensorflow as tf

from opts import Opts
from trainer import PGGAN
import dataset
import tests
import utils


def main(config):

    if config.task == 'prepare':
        print('Preparing ...')
        dataset.prepare_tf_record_dataset(
            dataset_dir=config.dataset_dir, 
            tf_record_save_dir=config.tf_record_save_dir, 
            dimensionality=config.dimensionality,
            glob_ext=config.glob_ext,
            n_img_per_shard=config.n_img_per_shard,
            start_resolution=config.start_resolution,
            target_resolution=config.target_resolution)

    elif config.task == 'train':
        print('Training ...')
        pggan = PGGAN(config)
        pggan.train()

    elif config.task == 'generate':
        print('Generating ...')
        utils.generate(config)

    elif config.task == 'test':
        print('Testing ...')
        tests.do_test(config)


if __name__ == '__main__':
    opts = Opts()
    config = opts.parse()
    main(config)