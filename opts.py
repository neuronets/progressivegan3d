import argparse
import os

import tensorflow as tf

class Opts:
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        self.subparsers = self.parser.add_subparsers(help='prepare | train | generate | test', dest='task')

        # Prepare Task
        self.parser_prepare = self.subparsers.add_parser('prepare', help='Prepare tf records dataset')

        self.parser_prepare.add_argument('--dataset_dir', required=True)
        self.parser_prepare.add_argument('--save_path', required=True)
        self.parser_prepare.add_argument('--dimensionality', default=2, type=int)

        # Train Task
        self.parser_train = self.subparsers.add_parser('train', help='Train the progressive GAN')

        self.parser_train.add_argument('--dataset', required=True, help='Training dataset')
        self.parser_train.add_argument('--run_id', default='1', help='Run ID to save data')
        self.parser_train.add_argument('--generated_dir', default='generated', help='Path in Run ID to store generated images')
        self.parser_train.add_argument('--model_dir', default='saved_models', help='Path in Run ID to store saved models')
        self.parser_train.add_argument('--log_dir', default='logs', help='Path in Run ID to store logs')

        self.parser_train.add_argument('--dimensionality', default=2, type=int, help='Dimensionality of models (2, 3)')
        self.parser_train.add_argument('--latent_size', default=256, type=int, help='Latent size for generator')
        self.parser_train.add_argument('--num_channels', default=1, type=int, help='Number of channels in images')
        self.parser_train.add_argument('--num_classes', default=1, type=int, help='Number of classes (only 1 supported')

        self.parser_train.add_argument('--kiters_per_transition', default=200, type=int, help='x*1000 iterations per transition')
        self.parser_train.add_argument('--kiters_per_resolution', default=200, type=int, help='x*1000 iterations per resolution')
        self.parser_train.add_argument('--start_resolution', default=4, type=int, help='start resolution')
        self.parser_train.add_argument('--target_resolution', default=256, type=int, help='target resolution')
        # self.parser_train.add_argument('--resolution_batch_size')
        self.parser_train.add_argument('--d_repeats', default=1, type=int, help='Batches of discriminator per generator batch')

        self.parser_train.add_argument('--lr', default=1e-4, type=float, help='learning rate')
        self.parser_train.add_argument('--gpus', default=['/gpu:0'], nargs='*', help='gpus to use')

        self.parser_train.add_argument('--d_fmap_base', default=2048, type=int, help='Discriminator fmap base')
        self.parser_train.add_argument('--g_fmap_base', default=2048, type=int, help='Generator fmap base')

        # Test Task
        self.parser_test = self.subparsers.add_parser('test', help='Tests to run on generator')

        self.parser_test.add_argument('--test_name', required=True, help='Current one of [interpolation | nearest_neighbor]')
        self.parser_test.add_argument('--model_file', required=True, help='Model file to saved generator h5 model')
        self.parser_test.add_argument('--save_dir', required=True, help='Directory to save test results')
        self.parser_test.add_argument('--tf_records_file', help='Dataset to check against for nearest_neighbor test')
        self.parser_test.add_argument('--latent_size', default=1024, type=int, help='Latent size for generator')
        self.parser_test.add_argument('--resolution', default=8, type=int)
        self.parser_test.add_argument('--dimensionality', default=3, type=int, help='Dimensionality of models (2, 3)')
        self.parser_test.add_argument('--n_interpolations', default=10, type=int, help='Number of interpolations for interpolationtest')


    def parse(self):
        config = self.parser.parse_args()

        if config.task=='train':

            if len(config.gpus)>1:
                config.strategy = tf.distribute.MirroredStrategy(devices=config.gpus)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = int(config.gpus[0][-1])
                config.gpus = '/gpu:0'
                config.strategy = None

            config.img_ext = 'jpg' if config.dimensionality == 2 else 'nii.gz'
            config.resolution_batch_size = {4: 64, 8: 32, 16: 16, 32: 8, 64: 4, 128: 1, 256: 1} # per gpu

            for res, batch_size in config.resolution_batch_size.items():
                config.resolution_batch_size[res] = batch_size * len(config.gpus)

        return config