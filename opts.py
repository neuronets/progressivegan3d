import argparse
import tensorflow as tf

class Opts:
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        self.subparsers = self.parser.add_subparsers(help='prepare | train | generate', dest='task')

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


    def parse(self):
        opt = self.parser.parse_args()

        if len(opt.gpus)>1:
            opt.strategy = tf.distribute.MirroredStrategy(devices=opt.gpus)

        opt.img_ext = 'jpg' if opt.dimensionality == 2 else 'nii.gz'
        opt.resolution_batch_size = {4: 64, 8: 32, 16: 16, 32: 8, 64: 4, 128: 2, 256: 1}

        return opt