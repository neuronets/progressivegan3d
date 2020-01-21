import os
import tensorflow as tf

from opts import Opts
from trainer import PGGAN
import dataset


def main(opts):

    if opts.task == 'prepare':
        dataset.prepare_tf_record_dataset(opts.dataset_dir, opts.save_path, opts.dimensionality)

    elif opts.task=='train':
        pggan = PGGAN(opts)
        pggan.train()


if __name__ == '__main__':
    opts = Opts()
    opts = opts.parse()
    main(opts)