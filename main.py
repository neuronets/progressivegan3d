import os
import tensorflow as tf

from opts import Opts
from trainer import PGGAN

# os.environ["CUDA_VISIBLE_DEVICES"] = "5"

def main(opts):
    # latents_size = 1024
    # n_classes = 1
    # dataset_dir = 'data/ixi_slices'
    # dataset_dir = 'data/ixi_t1'
    # dataset_dir = 'data/ixi_t1.tfrecord'

    # dimensionality = 3
    # gpus = ['/gpu:0', '/gpu:1', '/gpu:6', '/gpu:7']
    # # gpus = ['/gpu:0']
    # run_id = 'temp3d_1'

    if opts.task=='train':

        pggan = PGGAN(opts)
        pggan.train()


if __name__ == '__main__':
    opts = Opts()
    opts = opts.parse()
    main(opts)