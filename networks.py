from functools import partial

import tensorflow as tf
import tensorflow.keras.layers as layers


class Generator:
	'''
	Progressive Generator
	'''
	def __init__(self, latents_size, num_channels=3, fmap_base=8192, fmap_max=512):
		super(Generator, self).__init__()

		self.fmap_base = fmap_base
		self.fmap_max = fmap_max
		self.num_channels = num_channels

		self.latents_size = latents_size

		self.growing_generator = self._make_generator_base()
		self.train_generator = self.growing_generator

		self.current_resolution = 1

	def _pixel_norm(self, epsilon=1e-8):
		return layers.Lambda(lambda x: x * tf.math.rsqrt(tf.reduce_mean(tf.square(x), axis=1, keepdims=True) + epsilon))

	def _weighted_sum(self):
		# tf.print(inputs[0])
		# return layers.Lambda(lambda inputs : 0.5*inputs[0]+0.5*inputs[1])
		return layers.Lambda(lambda inputs : (1-inputs[2])*inputs[0] + (inputs[2])*inputs[1])

	def _nf(self, stage): 
		return min(int(self.fmap_base / (2.0 ** (stage))), self.fmap_max)

	def _make_generator_base(self):

		latents = layers.Input(shape=[self.latents_size], dtype=tf.float32, name='latents')
		alpha = layers.Input(shape=[], dtype=tf.float32, name='g_alpha')

		# Latents stage
		x = self._pixel_norm()(latents)
		x = layers.Dense(self._nf(1)*4)(x)
		x = layers.Reshape((2, 2, self._nf(1)))(x)

		return tf.keras.models.Model(inputs=[latents, alpha], outputs=[x], name='generator_base')

	def _make_generator_block(self, nf, name=''):
		block_layers = []

		block_layers.append(layers.Conv2DTranspose(nf, kernel_size=3, strides=2, padding='same'))
		block_layers.append(layers.Activation(tf.nn.leaky_relu))
		block_layers.append(self._pixel_norm())

		block_layers.append(layers.Conv2D(nf, kernel_size=3, strides=1, padding='same'))
		block_layers.append(layers.Activation(tf.nn.leaky_relu))
		block_layers.append(self._pixel_norm())

		return tf.keras.models.Sequential(block_layers, name=name)

	def add_resolution(self, res):

		g_block = self._make_generator_block(self._nf(res), name='g_block_{}'.format(res))

		g_block_output = g_block(self.growing_generator.output)

		# Residual from input
		to_rgb_1 = layers.UpSampling2D()(self.growing_generator.output)
		to_rgb_1 = layers.Conv2D(self.num_channels, kernel_size=1)(to_rgb_1)

		to_rgb_2 = layers.Conv2D(self.num_channels, kernel_size=1)(g_block_output)

		lerp_output = self._weighted_sum()([to_rgb_1, to_rgb_2, self.growing_generator.input[1]])

		self.growing_generator = tf.keras.models.Model(inputs=self.growing_generator.input, outputs=g_block_output)
		self.train_generator = tf.keras.models.Model(inputs=self.growing_generator.input, outputs=[lerp_output])
		self.current_resolution = res

	def get_current_resolution(self):
		return current_resolution

	def get_trainable_generator(self):
		return self.train_generator

	def get_inference_generator(self):
		raise NotImplementedError


class Discriminator:
	'''
	Progressive Discriminator
	'''

	def __init__(self, num_classes, num_channels=3, fmap_base=8192, fmap_max=512):
		super(Discriminator, self).__init__()

		self.fmap_base = fmap_base
		self.fmap_max = fmap_max
		self.num_channels = num_channels

		self.num_classes = num_classes

		self.growing_discriminator = self._make_discriminator_base()
		self.train_discriminator = self.growing_discriminator

		self.current_resolution = 1

	def _weighted_sum(self):
		return layers.Lambda(lambda inputs : 0.5*inputs[0]+0.5*inputs[1])

		return layers.Lambda(lambda inputs : (1-inputs[2])*inputs[0] + (inputs[2])*inputs[1])

	def _nf(self, stage): 
		return min(int(self.fmap_base / (2.0 ** (stage))), self.fmap_max)

	def _make_discriminator_base(self):

		inputs = layers.Input(shape=[2, 2, self._nf(1)], name='dummy')

		x = layers.Flatten()(inputs)
		x = layers.Dense(self._nf(1))(x)
		x = layers.Activation(tf.nn.leaky_relu)(x)

		output = layers.Dense(self.num_classes)(x)

		return tf.keras.models.Model(inputs=[inputs], outputs=output)

	def _make_discriminator_block(self, nf, name=''):
		block_layers = []

		block_layers.append(layers.Conv2D(nf, kernel_size=3, strides=1, padding='same'))
		block_layers.append(layers.Activation(tf.nn.leaky_relu))

		block_layers.append(layers.Conv2D(nf, kernel_size=3, strides=2, padding='same'))
		block_layers.append(layers.Activation(tf.nn.leaky_relu))

		return tf.keras.models.Sequential(block_layers, name=name)

	def add_resolution(self, res):

		inputs = layers.Input(shape=[2.0**res, 2.0**res, self.num_channels], name='image')
		alpha = layers.Input(shape=[], name='d_alpha')

		from_rgb_2 = layers.Conv2D(self._nf(res), kernel_size=1, padding='same', name='from_rgb_2')(inputs)

		d_block = self._make_discriminator_block(self._nf(res-1), name='d_block_{}'.format(res))

		from_rgb_2 = d_block(from_rgb_2)

		# Residual from input
		from_rgb_1 = layers.AveragePooling2D()(inputs)
		from_rgb_1 = layers.Conv2D(self._nf(res-1), kernel_size=1, padding='same', name='from_rgb_1')(from_rgb_1)

		lerp_input = self._weighted_sum()([from_rgb_1, from_rgb_2, alpha])

		score_output = self.growing_discriminator(lerp_input)

		self.growing_discriminator = tf.keras.Sequential([d_block, self.growing_discriminator])
		self.train_discriminator = tf.keras.models.Model(inputs=[inputs, alpha], outputs=[score_output])
		self.current_resolution = res

	def get_current_resolution(self):
		return self.current_resolution

	def get_trainable_discriminator(self):
		return self.train_discriminator

	def get_inference_discriminator(self):
		raise NotImplementedError





