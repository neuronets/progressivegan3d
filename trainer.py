import os
import numpy as np
from PIL import Image
import tensorflow as tf

import networks
import utils
import losses

class PGGAN(tf.Module):

    def __init__(
        self,
        latents_size,
        num_classes,
        dataset_dir,
        run_id = '1',
        log_dir = 'logs',
        generated_dir = 'generated',
        model_dir = 'saved_model',
        num_channels = 3,
        d_repeats = 4,
        iters_per_resolution = 300,
        iters_per_transition = 300,
        start_resolution = 4,
        target_resolution = 256,
        resolution_batch_size = {4: 64, 8: 64, 16: 64, 32: 32, 64: 16, 128: 8, 256: 8}):
        
        super(PGGAN, self).__init__()

        self.latents_size = latents_size
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.dataset_dir = dataset_dir
        self.d_repeats = d_repeats
        self.iters_per_resolution = iters_per_resolution*1000
        self.iters_per_transition = iters_per_transition*1000
        self.target_resolution = target_resolution
        self.resolution_batch_size = resolution_batch_size

        self.generator = networks.Generator(latents_size, num_channels=num_channels)
        self.discriminator = networks.Discriminator(num_classes, num_channels=num_channels)

        self.d_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        self.g_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

        self.run_id = run_id
        self.generated_dir = os.path.join(run_id, generated_dir)
        self.model_dir = os.path.join(run_id, model_dir)

        os.makedirs(self.run_id, exist_ok=True)
        os.makedirs(self.generated_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

        self.train_summary_writer = tf.summary.create_file_writer(os.path.join(log_dir, self.run_id))

        self.current_resolution = 2
        self.add_resolution()

        while 2**self.current_resolution<start_resolution:
            self.add_resolution()
            self.current_resolution+=1

    def get_g_train_step(self):
        '''
        tf function must be retraced for every growth of the model
        '''
        @tf.function()
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
        @tf.function()
        def d_train_step(latents, reals, alpha):
            with tf.GradientTape() as tape:
                fakes = self.train_generator([latents, alpha])

                fakes_pred = self.train_discriminator([fakes, alpha])
                reals_pred = self.train_discriminator([reals, alpha])

                w_real_loss = losses.wasserstein_loss(1, reals_pred)
                w_fake_loss = losses.wasserstein_loss(-1, fakes_pred) 

                average_samples = utils.random_weight_sample(reals, fakes)
                average_pred = self.train_discriminator([average_samples, alpha])

                average_gradients = tf.gradients(average_pred, average_samples)[0]

                gp_loss = losses.gradient_penalty_loss(average_gradients, 10)

                d_loss = w_real_loss + w_fake_loss + gp_loss

            d_gradients = tape.gradient(d_loss, self.train_discriminator.trainable_variables)

            self.d_optimizer.apply_gradients(zip(d_gradients, self.train_discriminator.trainable_variables))
            return d_loss
        return d_train_step

    def add_resolution(self):
        self.generator.add_resolution()
        self.discriminator.add_resolution()

        self.train_generator = self.generator.get_trainable_generator()
        self.train_discriminator = self.discriminator.get_trainable_discriminator()

    def get_current_batch_size(self):
        return self.resolution_batch_size[2**self.current_resolution]

    def get_current_dataset(self):
        batch_size = self.get_current_batch_size()
        dataset = utils.get_dataset(self.dataset_dir, self.current_resolution, 
                    batch_size, num_channels=self.num_channels)
        return dataset

    def get_current_alpha(self, iters_done):
        return iters_done/self.iters_per_transition

    def run_phase(self, phase):

        dataset = self.get_current_dataset()
        g_train_step = self.get_g_train_step()
        d_train_step = self.get_d_train_step()

        if phase == 'Transition':
            iters_total = self.iters_per_transition
            get_alpha = self.get_current_alpha
        else:
            iters_total = self.iters_per_resolution
            get_alpha = lambda x: 1.0

        g_loss_tracker = tf.keras.metrics.Mean()
        d_loss_tracker = tf.keras.metrics.Mean()

        iters_done = 0.0
        prog_bar = tf.keras.utils.Progbar(self.iters_per_transition, 
                stateful_metrics = ['Res', 'D Loss', 'G Loss'])

        while iters_done < iters_total:

            alpha = get_alpha(iters_done)
            batch_size = self.get_current_batch_size()

            prog_bar.add(0, [('Res', self.current_resolution-1+alpha)])
            alpha = tf.constant(alpha, tf.float32)

            i = 0
            for reals in dataset:
                iters_done+=batch_size

                if iters_done > self.iters_per_transition:
                    break

                latents = tf.random.normal((batch_size, self.latents_size)) 

                if i%self.d_repeats==0:
                    g_loss = g_train_step(latents, alpha)
                    g_loss_tracker.update_state(g_loss)
                
                d_loss = d_train_step(latents, reals, alpha)
                d_loss_tracker.update_state(d_loss)

                prog_bar.add(batch_size, [('G Loss', g_loss_tracker.result()), ('D Loss', d_loss_tracker.result())])
                i+=1

            if phase == 'Resolution':
                self.generate_samples(10)

        with self.train_summary_writer.as_default():
            tf.summary.scalar('G Loss', g_loss_tracker.result(), step=self.current_resolution)
            tf.summary.scalar('D Loss', d_loss_tracker.result(), step=self.current_resolution)

        g_loss_tracker.reset_states()
        d_loss_tracker.reset_states()

        self.save_models()
        print()

    def train(self):

        while 2**self.current_resolution < self.target_resolution:

            self.current_resolution += 1
            self.add_resolution() 

            print('Transition Phase')
            self.run_phase(phase='Transition')

            print('Resolution Phase')
            self.run_phase(phase='Resolution')
                
    def infer(self, latents):
        raise NotImplementedError

    def save_models(self):
        self.train_generator.save(os.path.join(self.model_dir, 'g_{}'.format(self.current_resolution)))
        self.train_discriminator.save(os.path.join(self.model_dir, 'd_{}'.format(self.current_resolution)))

    def generate_samples(self, num_samples):
        for i in range(num_samples):
            latents = tf.random.normal((1, self.latents_size)) 
            fakes = self.train_generator([latents, 1.0])
            fakes = utils.adjust_dynamic_range(fakes, [-1.0, 1.0], [0.0, 255.0])
            fakes = tf.clip_by_value(fakes, 0.0, 255.0)
            img_arr = np.squeeze(np.array(fakes[0])).astype(np.uint8)
            im = Image.fromarray(img_arr, 'L')
            im.save(os.path.join(self.generated_dir, 'res_{}_{}.jpg').format(self.current_resolution, i))







