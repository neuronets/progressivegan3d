import os
import tensorflow as tf

from opts import Opts
from trainer import PGGAN
import dataset
import tests


def main(config):

    if config.task == 'prepare':
        dataset.prepare_tf_record_dataset(
            dataset_dir=config.dataset_dir, 
            save_path=config.save_path, 
            dimensionaltiy=config.dimensionality)

    elif config.task == 'train':
        pggan = PGGAN(config)
        pggan.train()

    elif config.task == 'test':
        tests.do_test(config)


if __name__ == '__main__':
    opts = Opts()
    config = opts.parse()
    main(config)