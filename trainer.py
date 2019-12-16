import os
import tensorflow as tf

import networks
import utils

num_epochs = 50

start_resolution = 2
target_resolution = 256

resolution_batch_size = {4: 64, 8: 64, 16: 64, 32: 32, 64: 16, 128: 8, 256: 8}
epochs_per_resolution = 4
epochs_per_transition = 4

latents_size = 256
num_classes = 1
num_channels = 3

generator = networks.Generator(latents_size)
discriminator = networks.Discriminator(num_classes)
d_optimizer = tf.keras.optimizers.Adam()
g_optimizer = tf.keras.optimizers.Adam()
d_loss_tracker = tf.keras.metrics.Mean()
g_loss_tracker = tf.keras.metrics.Mean()

run_id = '1'
train_summary_writer = tf.summary.create_file_writer(os.path.join('./logs', run_id))
prog_bar = tf.keras.utils.Progbar(600, stateful_metrics = ['Epoch', 'Res', 'D Loss', 'G Loss'])

current_resolution = start_resolution 

for epoch in range(num_epochs):

	batch_size = resolution_batch_size[2**current_resolution]

	if epoch%(epochs_per_resolution+epochs_per_transition) == 0:
		generator.add_resolution(current_resolution)
		discriminator.add_resolution(current_resolution)

		train_generator = generator.get_trainable_generator()
		train_discriminator = discriminator.get_trainable_discriminator()

		# print(train_generator.summary())
		# print(train_discriminator.summary())

		dataset = utils.get_dataset('data/ixi_slices', current_resolution, 
			batch_size, num_channels=num_channels)

		current_resolution += 1
		epoch_in_resolution = 1

	if epoch_in_resolution > epochs_per_resolution:
		alpha = (epoch%epochs_per_transition)/epochs_per_transition
	else:
		alpha = 1

	prog_bar.update(0, [('Epoch', epoch+1)])
	prog_bar.update(0, [('Res', current_resolution+alpha)])

	for reals in dataset:
		latents = tf.random.normal((batch_size, latents_size))

		# Train G
		with tf.GradientTape() as tape:
			fakes = train_generator([latents, alpha])
			fakes_pred = train_discriminator([fakes, alpha])

			w_fake_loss = utils.wasserstein_loss(-1, fakes_pred) 
			g_loss = -1 * w_fake_loss
			g_loss_tracker.update_state(g_loss)

			g_gradients = tape.gradient(g_loss, train_generator.trainable_variables)

		g_optimizer.apply_gradients(zip(g_gradients, train_generator.trainable_variables))	

		# Train D
		with tf.GradientTape() as tape:
			fakes = train_generator([latents, alpha])

			fakes_pred = train_discriminator([fakes, alpha])
			reals_pred = train_discriminator([reals, alpha])

			w_real_loss = utils.wasserstein_loss(1, reals_pred)
			w_fake_loss = utils.wasserstein_loss(-1, fakes_pred) 
			d_loss = w_real_loss + w_fake_loss
			d_loss_tracker.update_state(d_loss)

			d_gradients = tape.gradient(d_loss, train_discriminator.trainable_variables)

		d_optimizer.apply_gradients(zip(d_gradients, train_discriminator.trainable_variables))

		prog_bar.add(batch_size, [('G Loss', g_loss_tracker.result()), ('D Loss', d_loss_tracker.result())])

	with train_summary_writer.as_default():
		tf.summary.scalar('G Loss', g_loss_tracker.result(), step=epoch)
		tf.summary.scalar('D Loss', d_loss_tracker.result(), step=epoch)

	g_loss_tracker.reset_states()
	d_loss_tracker.reset_states()

	epoch_in_resolution+=1






