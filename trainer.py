import os
from functools import partial
import numpy as np
from PIL import Image
import tensorflow as tf
import nibabel as nib

import networks
import utils
import losses

class PGGAN(tf.Module):

    def __init__(self, opts):
        
        super(PGGAN, self).__init__()

        self.dataset = opts.dataset
        self.latent_size = opts.latent_size
        self.num_classes = opts.num_classes
        self.dimensionality = opts.dimensionality
        self.num_channels = opts.num_channels
        self.learning_rate = opts.lr

        self.d_repeats = opts.d_repeats
        self.iters_per_resolution = opts.kiters_per_resolution*1000
        self.iters_per_transition = opts.kiters_per_transition*1000
        self.start_resolution = opts.start_resolution
        self.target_resolution = opts.target_resolution
        self.resolution_batch_size = opts.resolution_batch_size

        self.img_ext = opts.img_ext

        self.generator = networks.Generator(self.latent_size, dimensionality=self.dimensionality, 
            num_channels=self.num_channels, fmap_base=opts.g_fmap_base)
        self.discriminator = networks.Discriminator(self.num_classes, dimensionality=self.dimensionality, 
            num_channels=self.num_channels, fmap_base=opts.d_fmap_base)

        self.run_id = opts.run_id
        self.generated_dir = os.path.join(opts.run_id, opts.generated_dir)
        self.model_dir = os.path.join(opts.run_id, opts.model_dir)

        os.makedirs(self.run_id, exist_ok=True)
        os.makedirs(self.generated_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

        self.train_summary_writer = tf.summary.create_file_writer(os.path.join(self.run_id, opts.log_dir))

        current_resolution = 2
        self.add_resolution()

        while 2**current_resolution<self.start_resolution:
            self.add_resolution()
            current_resolution+=1

        self.strategy = opts.strategy
        if self.strategy is not None:
            with self.strategy.scope():

                self.generator = networks.Generator(self.latent_size, dimensionality=self.dimensionality, 
                    num_channels=self.num_channels, fmap_base=opts.g_fmap_base)
                self.discriminator = networks.Discriminator(self.num_classes, dimensionality=self.dimensionality, 
                    num_channels=self.num_channels, fmap_base=opts.d_fmap_base)

                current_resolution = 2
                self.add_resolution()

                while 2**current_resolution<self.start_resolution:
                    self.add_resolution()
                    current_resolution+=1

    # TODO: Write decorator
    def get_g_train_step(self):
        '''
        tf function must be retraced for every growth of the model
        '''
        @tf.function
        def g_train_step(latents, alpha):
            with tf.GradientTape() as tape:
                fakes = self.train_generator([latents, alpha])
                fakes_pred = self.train_discriminator([fakes, alpha])

                g_loss = losses.wasserstein_loss(1, fakes_pred) 

            g_gradients = tape.gradient(g_loss, self.train_generator.trainable_variables)
            self.g_optimizer.apply_gradients(zip(g_gradients, self.train_generator.trainable_variables))
            return g_loss
        return g_train_step

    def get_d_train_step(self):
        '''
        tf function must be retraced for every growth of the model
        '''
        @tf.function
        def d_train_step(latents, reals, alpha):
            with tf.GradientTape() as tape:
                fakes = self.train_generator([latents, alpha])

                fakes_pred = self.train_discriminator([fakes, alpha])
                reals_pred = self.train_discriminator([reals, alpha])

                w_real_loss = losses.wasserstein_loss(1, reals_pred)
                w_fake_loss = losses.wasserstein_loss(-1, fakes_pred) 

                average_samples = utils.random_weight_sample(reals, fakes)
                average_pred = self.train_discriminator([average_samples, alpha])

                gp_loss = losses.gradient_penalty_loss(average_pred, average_samples, weight=10)

                epsilon_loss = losses.epsilon_penalty_loss(reals_pred, weight=0.001)

                d_loss = w_real_loss + w_fake_loss + gp_loss + epsilon_loss

            d_gradients = tape.gradient(d_loss, self.train_discriminator.trainable_variables)
            self.d_optimizer.apply_gradients(zip(d_gradients, self.train_discriminator.trainable_variables))

            return d_loss
        return d_train_step

    def get_mirrored_g_train_step(self, global_batch_size):
        @tf.function
        def g_train_step(latents, alpha, global_batch_size=1):
            def step_fn(latents, alpha):
                with tf.GradientTape() as tape:
                    latents = tf.random.normal((global_batch_size//4, self.latent_size))

                    fakes = self.train_generator([latents, alpha])
                    fakes_pred = self.train_discriminator([fakes, alpha])
                    g_loss = losses.wasserstein_loss(1, fakes_pred, reduction=False) 
                    g_gradient_loss = tf.reduce_sum(g_loss) * 1/global_batch_size

                g_gradients = tape.gradient(g_gradient_loss, self.train_generator.trainable_variables)
                self.g_optimizer.apply_gradients(zip(g_gradients, self.train_generator.trainable_variables))
                return g_loss

            per_example_losses = self.strategy.experimental_run_v2(step_fn, args=(latents, alpha))
            mean_loss = self.strategy.reduce(
                tf.distribute.ReduceOp.SUM, per_example_losses, axis=0)
            return mean_loss/global_batch_size
        return partial(g_train_step, global_batch_size=global_batch_size)

    def get_mirrored_d_train_step(self, global_batch_size):
        @tf.function
        def d_train_step(latents, reals, alpha, global_batch_size=1):
            def step_fn(latents, reals, alpha):
                with tf.GradientTape() as tape:
                    latents = tf.random.normal((global_batch_size//4, self.latent_size))
                    fakes = self.train_generator([latents, alpha])

                    fakes_pred = self.train_discriminator([fakes, alpha])
                    reals_pred = self.train_discriminator([reals, alpha])

                    w_real_loss = losses.wasserstein_loss(1, reals_pred, reduction=False)
                    w_fake_loss = losses.wasserstein_loss(-1, fakes_pred, reduction=False) 

                    average_samples = utils.random_weight_sample(reals, fakes)
                    average_pred = self.train_discriminator([average_samples, alpha])

                    gp_loss = losses.gradient_penalty_loss(average_pred, average_samples, weight=10, reduction=False)

                    epsilon_loss = losses.epsilon_penalty_loss(reals_pred, weight=0.001)

                    d_loss = w_real_loss + w_fake_loss + gp_loss + epsilon_loss
                    d_gradient_loss = d_loss * (1/global_batch_size)

                d_gradients = tape.gradient(d_gradient_loss, self.train_discriminator.trainable_variables)
                self.d_optimizer.apply_gradients(zip(d_gradients, self.train_discriminator.trainable_variables))

                return d_loss

            per_example_losses = self.strategy.experimental_run_v2(step_fn, args=(latents, reals, alpha))
            mean_loss = self.strategy.reduce(
                tf.distribute.ReduceOp.SUM, per_example_losses, axis=0)
            return mean_loss/global_batch_size
        return partial(d_train_step, global_batch_size=global_batch_size)


    def add_resolution(self):
        self.generator.add_resolution()
        self.discriminator.add_resolution()

        self.train_generator = self.generator.get_trainable_generator()
        self.train_discriminator = self.discriminator.get_trainable_discriminator()

    def get_current_batch_size(self, current_resolution):
        return self.resolution_batch_size[2**current_resolution]

    def get_current_dataset(self, current_resolution):
        batch_size = self.get_current_batch_size(current_resolution)
        if self.dimensionality == 2:
            dataset = utils.get_dataset(self.dataset, current_resolution, 
                        batch_size, dimensionality=self.dimensionality, num_channels=self.num_channels, img_ext=self.img_ext)
        else:
            dataset = utils.get_dataset_3d(self.dataset, current_resolution, batch_size)
        return dataset

    def get_current_alpha(self, iters_done):
        return iters_done/self.iters_per_transition

    def run_phase(self, phase, current_resolution):

        dataset = self.get_current_dataset(current_resolution)
        g_train_step = self.get_g_train_step()
        d_train_step = self.get_d_train_step()

        self.d_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=0.0, beta_2=0.99, epsilon=1e-8)
        self.g_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=0.0, beta_2=0.99, epsilon=1e-8)

        if self.strategy is not None:
            with self.strategy.scope():
                dataset = self.strategy.experimental_distribute_dataset(dataset)
                g_train_step = self.get_mirrored_g_train_step(self.get_current_batch_size(current_resolution))
                d_train_step = self.get_mirrored_d_train_step(self.get_current_batch_size(current_resolution))
                self.d_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=0.0, beta_2=0.99, epsilon=1e-8)
                self.g_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=0.0, beta_2=0.99, epsilon=1e-8)

        if phase == 'Transition':
            iters_total = self.iters_per_transition
            get_alpha = self.get_current_alpha
        else:
            iters_total = self.iters_per_resolution
            get_alpha = lambda x: 1.0

        g_loss_tracker = tf.keras.metrics.Mean()
        d_loss_tracker = tf.keras.metrics.Mean()

        iters_done = 0
        prog_bar = tf.keras.utils.Progbar(self.iters_per_transition, 
                stateful_metrics = ['Res', 'D Loss', 'G Loss'])

        while iters_done < iters_total:

            alpha = get_alpha(iters_done)
            batch_size = self.get_current_batch_size(current_resolution)

            prog_bar.add(0, [('Res', current_resolution-1+alpha)])
            alpha = tf.constant(alpha, tf.float32)

            i = 0
            # with self.strategy.scope():

            for reals in dataset:
                iters_done+=batch_size

                if iters_done > self.iters_per_transition:
                    break

                with self.strategy.scope():
                    latents = tf.random.normal((batch_size, self.latent_size)) 
                # print(latents)

                if i%self.d_repeats==0:
                    with self.strategy.scope():
                        g_loss = g_train_step(latents, alpha)
                    g_loss_tracker.update_state(g_loss)
                
                with self.strategy.scope():
                    d_loss = d_train_step(latents, reals, alpha)
                d_loss_tracker.update_state(d_loss)

                prog_bar.add(batch_size, [('G Loss', g_loss_tracker.result()), ('D Loss', d_loss_tracker.result())])
                i+=1

            if phase == 'Resolution':
                self.generate_samples_3d(10, current_resolution)

            with self.train_summary_writer.as_default():
                tf.summary.scalar('G Loss {} {}'.format(phase, current_resolution), g_loss_tracker.result(), step=iters_done)
                tf.summary.scalar('D Loss {} {}'.format(phase, current_resolution), d_loss_tracker.result(), step=iters_done)

        g_loss_tracker.reset_states()
        d_loss_tracker.reset_states()

        self.save_models(current_resolution)
        print()

    def train(self):

        current_resolution = int(np.log2(self.start_resolution))
        while 2**current_resolution < self.target_resolution:
            with self.strategy.scope():
                current_resolution += 1
                self.add_resolution() 

            print('Transition Phase')
            self.run_phase(phase='Transition', current_resolution=current_resolution)

            print('Resolution Phase')
            self.run_phase(phase='Resolution', current_resolution=current_resolution)
                
    def infer(self, latents):
        raise NotImplementedError

    def save_models(self, current_resolution):
        self.train_generator.save(os.path.join(self.model_dir, 'g_{}.h5'.format(current_resolution)))
        self.train_discriminator.save(os.path.join(self.model_dir, 'd_{}.h5'.format(current_resolution)))

    def generate_samples(self, num_samples, current_resolution):
        for i in range(num_samples):
            latents = tf.random.normal((1, self.latent_size)) 
            fakes = self.train_generator([latents, 1.0])
            fakes = utils.adjust_dynamic_range(fakes, [-1.0, 1.0], [0.0, 255.0])
            fakes = tf.clip_by_value(fakes, 0.0, 255.0)
            img_arr = np.squeeze(np.array(fakes[0])).astype(np.uint8)
            im = Image.fromarray(img_arr, 'L')
            im.save(os.path.join(self.generated_dir, 'res_{}_{}.jpg').format(current_resolution, i))

    def generate_samples_3d(self, num_samples, current_resolution):
        for i in range(num_samples):
            latents = tf.random.normal((1, self.latent_size)) 
            fakes = self.train_generator([latents, 1.0])
            fakes = utils.adjust_dynamic_range(fakes, [-1.0, 1.0], [0.0, 255.0])
            fakes = tf.clip_by_value(fakes, 0.0, 255.0)
            img_arr = np.squeeze(np.array(fakes[0])).astype(np.uint8)
            mri = nib.Nifti1Image(img_arr, np.eye(4))
            nib.save(mri, os.path.join(self.generated_dir, 'res_{}_{}.nii.gz').format(current_resolution, i))







