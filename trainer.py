import os
import tensorflow as tf

import networks
import utils

class PGGAN(tf.Module):

    def __init__(
        self,
        latents_size,
        num_classes,
        dataset_dir,
        log_dir = './logs',
        run_id = '1',
        num_channels = 3,
        iters_per_epoch = {4: 10, 8: 10, 16: 10, 32: 10, 64: 10, 128: 10, 256: 10},
        iters_per_resolution = 20,
        iters_per_transition = 20,
        target_resolution = 256,
        resolution_batch_size = {4: 64, 8: 64, 16: 64, 32: 32, 64: 16, 128: 8, 256: 8}):
        
        super(PGGAN, self).__init__()

        self.latents_size = latents_size
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.dataset_dir = dataset_dir
        self.iters_per_resolution = iters_per_resolution*1000
        self.iters_per_transition = iters_per_transition*1000
        self.target_resolution = target_resolution
        self.resolution_batch_size = resolution_batch_size

        self.generator = networks.Generator(latents_size, num_channels=num_channels)
        self.discriminator = networks.Discriminator(num_classes, num_channels=num_channels)

        self.d_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        self.g_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

        self.run_id = run_id

        self.train_summary_writer = tf.summary.create_file_writer(os.path.join(log_dir, self.run_id))

        self.current_resolution = 1

    def get_g_train_step(self):
        '''
        tf function must be retraced for every growth of the model
        '''
        @tf.function
        def g_train_step(latents, alpha):
            with tf.GradientTape() as tape:
                fakes = self.train_generator([latents, alpha])
                fakes_pred = self.train_discriminator([fakes, alpha])

                w_fake_loss = utils.wasserstein_loss(-1, fakes_pred) 
                g_loss = -1 * w_fake_loss

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

                w_real_loss = utils.wasserstein_loss(1, reals_pred)
                w_fake_loss = utils.wasserstein_loss(-1, fakes_pred) 
                d_loss = w_real_loss + w_fake_loss

            d_gradients = tape.gradient(d_loss, self.train_discriminator.trainable_variables)

            self.d_optimizer.apply_gradients(zip(d_gradients, self.train_discriminator.trainable_variables))
            return d_loss
        return d_train_step

    # @tf.function
    def add_resolution(self):

        print('Adding resolution')

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

    def get_alpha(self, iters_done):
        return (iters_done%self.iters_per_transition)/self.iters_per_transition

    def train(self):

        g_loss_tracker = tf.keras.metrics.Mean()
        d_loss_tracker = tf.keras.metrics.Mean()

        while self.current_resolution < self.target_resolution:
            self.current_resolution += 1
            self.add_resolution()
            dataset = self.get_current_dataset()

            g_train_step = self.get_g_train_step()
            d_train_step = self.get_d_train_step()

            # Transition phase

            # iters_done = 0 

            # while iters_done < self.iters_per_transition:

            #     prog_bar = tf.keras.utils.Progbar(self.iters_per_transition, 
            #         stateful_metrics = ['Res', 'D Loss', 'G Loss'])

            #     alpha = self.get_alpha(iters_done)
            #     batch_size = self.get_current_batch_size()

            #     prog_bar.update(0, [('Res', self.current_resolution-1+alpha)])

            #     i = 0
            #     for reals in dataset:
            #         iters_done+=batch_size
            #         latents = tf.random.normal((batch_size, self.latents_size)) 

            #         if i%4==0:
            #             g_loss = self.g_train_step(latents, alpha)
            #             g_loss_tracker.update_state(g_loss)
                    
            #         d_loss = self.d_train_step(latents, reals, alpha)
            #         d_loss_tracker.update_state(d_loss)

            #         prog_bar.add(batch_size, [('G Loss', g_loss_tracker.result()), ('D Loss', d_loss_tracker.result())])

            # with self.train_summary_writer.as_default():
            #     tf.summary.scalar('G Loss', g_loss_tracker.result(), step=iters_done)
            #     tf.summary.scalar('D Loss', d_loss_tracker.result(), step=iters_done)

            # g_loss_tracker.reset_states()
            # d_loss_tracker.reset_states()            

            # Resolution phase

            iters_done = 0 

            prog_bar = tf.keras.utils.Progbar(self.iters_per_resolution, 
                    stateful_metrics = ['Res', 'D Loss', 'G Loss'])

            prog_bar.update(0, [('Res', self.current_resolution)])

            while iters_done < self.iters_per_resolution:

                alpha = 1.0
                batch_size = self.get_current_batch_size()

                i = 0
                for reals in dataset:
                    iters_done+=batch_size

                    if iters_done > self.iters_per_resolution:
                        break

                    latents = tf.random.normal((batch_size, self.latents_size)) 

                    if i%4==0:
                        g_loss = g_train_step(latents, alpha)
                        g_loss_tracker.update_state(g_loss)

                    i+=1
                    
                    d_loss = d_train_step(latents, reals, alpha)
                    d_loss_tracker.update_state(d_loss)

                    prog_bar.add(batch_size, [('G Loss', g_loss_tracker.result()), ('D Loss', d_loss_tracker.result())])

            print()
            with self.train_summary_writer.as_default():
                tf.summary.scalar('G Loss', g_loss_tracker.result(), step=iters_done)
                tf.summary.scalar('D Loss', d_loss_tracker.result(), step=iters_done)

            g_loss_tracker.reset_states()
            d_loss_tracker.reset_states()

                
    def infer(self, latents):
        raise NotImplementedError

    def save(self, save_dir):
        self.train_generator.save(os.path,join(save_dir, 'g_{}'.format(self.current_resolution)))
        self.train_discriminator.save(os.path,join(save_dir, 'd_{}'.format(self.current_resolution)))







