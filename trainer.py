from pathlib import Path
from functools import partial
import time

import numpy as np
from PIL import Image
import tensorflow as tf

# tf.debugging.set_log_device_placement(True)
# policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
# tf.keras.mixed_precision.experimental.set_policy(policy)

import nibabel as nib

import networks
import utils
import losses

class PGGAN(tf.Module):

    def __init__(self, config):
        
        super(PGGAN, self).__init__()

        self.tf_record_dir = config.tf_record_dir
        self.latent_size = config.latent_size
        self.label_size = config.label_size
        self.labels_exist = self.label_size > 0

        self.dimensionality = config.dimensionality
        self.num_channels = config.num_channels
        self.learning_rate = config.lr
        self.gpus = config.gpus

        self.d_repeats = config.d_repeats

        self.iters_per_transition = config.kiters_per_transition
        self.iters_per_resolution = config.kiters_per_resolution

        self.start_resolution = config.start_resolution
        self.target_resolution = config.target_resolution
        self.resolution_batch_size = config.resolution_batch_size

        self.img_ext = config.img_ext

        self.generator = networks.Generator(self.latent_size, dimensionality=self.dimensionality, 
            num_channels=self.num_channels, fmap_base=config.g_fmap_base)
        self.discriminator = networks.Discriminator(self.label_size, dimensionality=self.dimensionality, 
            num_channels=self.num_channels, fmap_base=config.d_fmap_base)

        self.run_id = Path(config.run_id)
        self.generated_dir = self.run_id.joinpath(config.generated_dir)
        self.model_dir = self.run_id.joinpath(config.model_dir)
        self.log_dir = self.run_id.joinpath(config.log_dir)

        self.run_id.mkdir(exist_ok=True)
        self.generated_dir.mkdir(exist_ok=True)
        self.model_dir.mkdir(exist_ok=True)

        self.train_summary_writer = tf.summary.create_file_writer(str(self.log_dir))

        current_resolution = 2
        self.add_resolution()

        while 2**current_resolution<self.start_resolution:
            self.add_resolution()
            current_resolution+=1

        self.strategy = config.strategy
        if self.strategy is not None:
            with self.strategy.scope():

                self.generator = networks.Generator(self.latent_size, dimensionality=self.dimensionality, 
                    num_channels=self.num_channels, fmap_base=config.g_fmap_base)
                self.discriminator = networks.Discriminator(self.label_size, dimensionality=self.dimensionality, 
                    num_channels=self.num_channels, fmap_base=config.d_fmap_base)

                current_resolution = 2
                self.add_resolution()

                while 2**current_resolution<self.start_resolution:
                    self.add_resolution()
                    current_resolution+=1

                self.generator.train_generator.load_weights(str(self.model_dir.joinpath('g_{}.h5'.format(current_resolution))))
                self.discriminator.train_discriminator.load_weights(str(self.model_dir.joinpath('d_{}.h5'.format(current_resolution))))


    # TODO: Write decorator
    def get_g_train_step(self):
        '''
        tf function must be retraced for every growth of the model
        '''
        @tf.function
        def g_train_step(latents, labels, alpha):
            with tf.GradientTape() as tape:
                fakes = self.train_generator([latents, alpha])
                fakes_pred, labels_pred = self.train_discriminator([fakes, alpha])

                g_loss = losses.wasserstein_loss(1, fakes_pred) 
                # scaled_loss = self.g_optimizer.get_scaled_loss(g_loss)

                if self.label_size>0:
                    g_loss += losses.labels_loss(labels, labels_pred, self.label_size)

            g_gradients = tape.gradient(g_loss, self.train_generator.trainable_variables)
            # scaled_gradients = tape.gradient(scaled_loss, self.train_generator.trainable_variables)
            # g_gradients = self.g_optimizer.get_unscaled_gradients(scaled_gradients)
            self.g_optimizer.apply_gradients(zip(g_gradients, self.train_generator.trainable_variables))
            return g_loss
        return g_train_step

    def get_d_train_step(self):
        '''
        tf function must be retraced for every growth of the model
        '''
        @tf.function
        def d_train_step(latents, reals, labels, alpha):
            with tf.GradientTape() as tape:
                fakes = self.train_generator([latents, alpha])

                fakes_pred, labels_pred_fake = self.train_discriminator([fakes, alpha])
                reals_pred, labels_pred_real = self.train_discriminator([reals, alpha])

                w_fake_loss = losses.wasserstein_loss(-1, fakes_pred)
                w_real_loss = losses.wasserstein_loss(1, reals_pred)

                average_samples = utils.random_weight_sample(reals, fakes, dimensionality=self.dimensionality)
                average_pred = self.train_discriminator([average_samples, alpha])

                gp_loss = losses.gradient_penalty_loss(average_pred, average_samples, weight=10)

                epsilon_loss = losses.epsilon_penalty_loss(reals_pred, weight=0.001)

                d_loss = w_real_loss + w_fake_loss + gp_loss + epsilon_loss
                # scaled_loss = self.d_optimizer.get_scaled_loss(d_loss)

                if self.label_size>0:
                    d_loss += losses.labels_loss(labels, labels_pred_fake, self.label_size, weight=1.0)
                    d_loss += losses.labels_loss(labels, labels_pred_real, self.label_size, weight=1.0)

            d_gradients = tape.gradient(d_loss, self.train_discriminator.trainable_variables)
            # scaled_gradients = tape.gradient(scaled_loss, self.train_discriminator.trainable_variables)
            # d_gradients = self.d_optimizer.get_unscaled_gradients(scaled_gradients)
            self.d_optimizer.apply_gradients(zip(d_gradients, self.train_discriminator.trainable_variables))

            return d_loss
        return d_train_step

    def get_mirrored_g_train_step(self, global_batch_size):
        @tf.function
        def g_train_step(latents, labels, alpha, global_batch_size=1):
            def step_fn(latents, labels, alpha):
                with tf.GradientTape() as tape:
                    # latents = self.sample_random_latents(global_batch_size//len(self.gpus))
                    latents = tf.random.normal((global_batch_size//len(self.gpus), self.latent_size))

                    fakes = self.train_generator([latents, alpha])
                    fakes_pred, labels_pred = self.train_discriminator([fakes, alpha])

                    g_loss = losses.wasserstein_loss(1, fakes_pred, reduction=False) 
                    if self.label_size>0:
                        g_loss += losses.labels_loss(labels, labels_pred, self.label_size, weight=1.0)

                    g_gradient_loss = tf.reduce_sum(g_loss) * 1/global_batch_size

                g_gradients = tape.gradient(g_gradient_loss, self.train_generator.trainable_variables)
                self.g_optimizer.apply_gradients(zip(g_gradients, self.train_generator.trainable_variables))
                return g_loss

            per_example_losses = self.strategy.experimental_run_v2(step_fn, args=(latents, labels, alpha))
            mean_loss = self.strategy.reduce(
                tf.distribute.ReduceOp.SUM, per_example_losses, axis=0)
            return mean_loss/global_batch_size
        return partial(g_train_step, global_batch_size=global_batch_size)

    def get_mirrored_d_train_step(self, global_batch_size):
        @tf.function
        def d_train_step(latents, reals, labels, alpha, global_batch_size=1):
            def step_fn(latents, reals, labels, alpha):
                with tf.GradientTape() as tape:

                    # latents = self.sample_random_latents(global_batch_size//len(self.gpus))
                    latents = tf.random.normal((global_batch_size//len(self.gpus), self.latent_size))

                    fakes = self.train_generator([latents, alpha])
                    fakes_pred, labels_pred_fake = self.train_discriminator([fakes, alpha])
                    reals_pred, labels_pred_real = self.train_discriminator([reals, alpha])

                    w_real_loss = losses.wasserstein_loss(1, reals_pred, reduction=False)
                    w_fake_loss = losses.wasserstein_loss(-1, fakes_pred, reduction=False) 

                    average_samples = utils.random_weight_sample(reals, fakes, self.dimensionality)
                    average_pred = self.train_discriminator([average_samples, alpha])

                    gp_loss = losses.gradient_penalty_loss(average_pred, average_samples, weight=10, reduction=False)

                    epsilon_loss = losses.epsilon_penalty_loss(reals_pred, weight=0.001)

                    d_loss = w_real_loss + w_fake_loss + gp_loss + epsilon_loss

                    if self.label_size>0:
                        d_loss += losses.labels_loss(labels, labels_pred_fake, self.label_size, weight=1.0)
                        d_loss += losses.labels_loss(labels, labels_pred_real, self.label_size, weight=1.0)

                    d_gradient_loss = d_loss * (1/global_batch_size)

                d_gradients = tape.gradient(d_gradient_loss, self.train_discriminator.trainable_variables)
                self.d_optimizer.apply_gradients(zip(d_gradients, self.train_discriminator.trainable_variables))

                return d_loss

            per_example_losses = self.strategy.experimental_run_v2(step_fn, args=(latents, reals, labels, alpha))
            mean_loss = self.strategy.reduce(
                tf.distribute.ReduceOp.SUM, per_example_losses, axis=0)
            return mean_loss/global_batch_size
        return partial(d_train_step, global_batch_size=global_batch_size)


    def add_resolution(self):
        self.generator.add_resolution()
        self.discriminator.add_resolution()

        self.train_generator = self.generator.get_trainable_generator()
        self.train_discriminator = self.discriminator.get_trainable_discriminator()


    def sample_random_latents(self, batch_size, label=None):
        latents = tf.random.normal((batch_size, self.latent_size - self.label_size))
        if self.labels_exist:
            if label is None:
                labels = tf.random.uniform((batch_size,), maxval=self.label_size, dtype=tf.int32)
            else:
                labels = [label]*batch_size
            labels = tf.one_hot(labels, depth=self.label_size)
            latents = tf.concat((latents, labels), axis=1)
        return latents

    def get_current_batch_size(self, current_resolution):
        return self.resolution_batch_size[2**current_resolution]

    def get_current_dataset(self, current_resolution):
        batch_size = self.get_current_batch_size(current_resolution)
        dataset = utils.get_dataset(self.tf_record_dir, current_resolution, batch_size, self.dimensionality, self.labels_exist)
        return dataset

    def get_current_alpha(self, iters_done, iters_per_transition):
        return iters_done/iters_per_transition

    def run_phase(self, phase, current_resolution):

        start = time.time()

        dataset = self.get_current_dataset(current_resolution)
        g_train_step = self.get_g_train_step()
        d_train_step = self.get_d_train_step()

        self.d_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=0.0, beta_2=0.99, epsilon=1e-8)
        # self.d_optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(self.d_optimizer , loss_scale='dynamic')

        self.g_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=0.0, beta_2=0.99, epsilon=1e-8)
        # self.g_optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(self.g_optimizer, loss_scale='dynamic')

        if self.strategy is not None:
            with self.strategy.scope():
                dataset = self.strategy.experimental_distribute_dataset(dataset)
                g_train_step = self.get_mirrored_g_train_step(self.get_current_batch_size(current_resolution))
                d_train_step = self.get_mirrored_d_train_step(self.get_current_batch_size(current_resolution))
                self.d_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=0.0, beta_2=0.99, epsilon=1e-8)
                self.g_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=0.0, beta_2=0.99, epsilon=1e-8)

        if phase == 'Transition':
            iters_total = self.iters_per_transition[2**current_resolution]
            get_alpha = self.get_current_alpha
        else:
            iters_total = self.iters_per_resolution[2**current_resolution]
            get_alpha = lambda x, y: 1.0

        g_loss_tracker = tf.keras.metrics.Mean()
        d_loss_tracker = tf.keras.metrics.Mean()

        iters_done = 0
        prog_bar = tf.keras.utils.Progbar(iters_total, verbose = 1,
                stateful_metrics = ['Res', 'D Loss', 'G Loss'])

        while iters_done < iters_total:

            alpha = get_alpha(iters_done, iters_total)
            batch_size = self.get_current_batch_size(current_resolution)

            prog_bar.add(0, [('Res', current_resolution-1+alpha)])
            alpha = tf.constant(alpha, tf.float32)

            i = 0

            for reals, labels in dataset:
                iters_done+=batch_size

                if iters_done > iters_total:
                    break

                latents = self.sample_random_latents(batch_size)

                if i%self.d_repeats==0:
                    if self.strategy is not None:
                        with self.strategy.scope():
                            g_loss = g_train_step(latents, labels, alpha)
                    else:
                        g_loss = g_train_step(latents, labels, alpha)
                    g_loss_tracker.update_state(g_loss)
                
                if self.strategy is not None:
                    with self.strategy.scope():
                        d_loss = d_train_step(latents, reals, labels, alpha)
                else:
                    d_loss = d_train_step(latents, reals, labels, alpha)
                d_loss_tracker.update_state(d_loss)

                prog_bar.add(batch_size, [('G Loss', g_loss_tracker.result()), ('D Loss', d_loss_tracker.result())])
                i+=1

            if phase == 'Resolution':
                self.generate_samples_3d(10, current_resolution)

            with self.train_summary_writer.as_default(), tf.name_scope(phase) as scope:
                tf.summary.scalar('G Loss {}'.format(current_resolution), g_loss_tracker.result(), step=iters_done)
                tf.summary.scalar('D Loss {}'.format(current_resolution), d_loss_tracker.result(), step=iters_done)

        g_loss_tracker.reset_states()
        d_loss_tracker.reset_states()

        self.save_models(current_resolution)
        print()
        print('Time taken : {}'.format(time.time()-start))

    def train(self):

        current_resolution = int(np.log2(self.start_resolution))
        while 2**current_resolution < self.target_resolution:
            if self.strategy is not None:
                with self.strategy.scope():
                    current_resolution += 1
                    self.add_resolution() 
            else:
                current_resolution += 1
                self.add_resolution() 

            print('Transition Phase')
            self.run_phase(phase='Transition', current_resolution=current_resolution)

            print('Resolution Phase')
            self.run_phase(phase='Resolution', current_resolution=current_resolution)
                
    def infer(self, latents):
        raise NotImplementedError

    def save_models(self, current_resolution):
        self.train_generator.save(str(self.model_dir.joinpath('g_{}.h5'.format(current_resolution))))
        self.train_discriminator.save(str(self.model_dir.joinpath('d_{}.h5'.format(current_resolution))))

    def generate_samples(self, num_samples, current_resolution):
        for i in range(num_samples):
            latents = tf.random.normal((1, self.latent_size)) 
            fakes = self.train_generator([latents, 1.0])
            fakes = utils.adjust_dynamic_range(fakes, [-1.0, 1.0], [0.0, 255.0])
            fakes = tf.clip_by_value(fakes, 0.0, 255.0)
            img_arr = np.squeeze(np.array(fakes[0])).astype(np.uint8)
            im = Image.fromarray(img_arr, 'L')
            im.save(str(self.generated_dir.joinpath('res_{}_{}.jpg'.format(current_resolution, i))))

    def generate_samples_3d(self, num_samples, current_resolution):
        for i in range(num_samples):
            latents = self.sample_random_latents(batch_size=1, label=i%2)
            fakes = self.train_generator([latents, 1.0])
            fakes = utils.adjust_dynamic_range(fakes, [-1.0, 1.0], [0.0, 255.0])
            fakes = tf.clip_by_value(fakes, 0.0, 255.0)
            img_arr = np.squeeze(np.array(fakes[0])).astype(np.uint8)
            mri = nib.Nifti1Image(img_arr, np.eye(4))
            nib.save(mri, str(self.generated_dir.joinpath('res_{}_{}.nii.gz'.format(current_resolution, i))))







