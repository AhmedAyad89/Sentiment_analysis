from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from utils import *

class autoencoder():
	def __init__(self, data, num_step=3000):
		# Training Parameters
		self.learning_rate = 0.008
		self.num_steps = num_step
		batch_size = data.batch_size

		self.display_step = 200

		# Network Parameters
		num_hidden_1 = 256 # 1st layer num features
		num_hidden_2 = 128 # 2nd layer num features (the latent dim)
		self.num_hidden_final =128
		num_input = data.data_dim

		self.data = data

		# tf Graph input
		self.X = tf.placeholder("float", [None, num_input])
		self.Y = tf.placeholder("float", [None, self.num_hidden_final])
		self.weights = {
				'encoder_h1': tf.Variable(tf.random_normal([num_input, num_hidden_1])),
				'encoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2])),
				'decoder_h1': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_1])),
				'decoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_input])),
		}
		self.biases = {
				'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
				'encoder_b2': tf.Variable(tf.random_normal([num_hidden_2])),
				'decoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
				'decoder_b2': tf.Variable(tf.random_normal([num_input])),
		}
		def build_model(self):
			# Building the encoder
			def encoder(x):
					# Encoder Hidden layer with sigmoid activation #1
					layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weights['encoder_h1']),
																				 self.biases['encoder_b1']))
					# Encoder Hidden layer with sigmoid activation #2
					layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, self.weights['encoder_h2']),
																				 self.biases['encoder_b2']))
					return layer_2

			# Building the decoder
			def decoder(x):
					# Decoder Hidden layer with sigmoid activation #1
					layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weights['decoder_h1']),
																				 self.biases['decoder_b1']))
					# Decoder Hidden layer with sigmoid activation #2
					layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, self.weights['decoder_h2']),
																				 self.biases['decoder_b2']))
					return layer_2

			# Construct model
			self.encoder_op = encoder(self.X)
			self.decoder_op = decoder(self.encoder_op)

			# Prediction
			y_pred = self.decoder_op
			# Targets (Labels) are the input data.
			y_true = self.X

			# Define loss and optimizer, minimize the squared error
			self.loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
			self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

			#define latent op
			self.latent = encoder(self.X)
			self.decoded = decoder(self.Y)
		build_model(self)
		self.sess = tf.Session()

	def train(self):
		# Initialize the variables (i.e. assign their default value)
		init = tf.global_variables_initializer()
		# Run the initializer
		self.sess.run(init)
		# Training
		for i in range(1, self.num_steps+1):
				# Prepare Data
				# Get the next batch of MNIST data (only images are needed, not labels)
				batch_x = self.data.next()

				# Run optimization op (backprop) and cost op (to get loss value)
				_, l = self.sess.run([self.optimizer, self.loss], feed_dict={self.X: batch_x})
				# Display logs per step
				if i % self.display_step == 0 or i == 1:
						print('Step %i: Minibatch Loss: %f' % (i, l))

	def test(self):
		# Testing
		# Encode and decode images from test set and visualize their reconstruction.
		n = 4
		canvas_orig = np.empty((28 * n, 28 * n))
		canvas_recon = np.empty((28 * n, 28 * n))
		for i in range(n):
				# MNIST test set
				batch_x, _ = mnist.test.next_batch(n)
				# Encode and decode the digit image
				g = sess.run(decoder_op, feed_dict={X: batch_x})

				# Display original images
				for j in range(n):
						# Draw the original digits
						canvas_orig[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
								batch_x[j].reshape([28, 28])
				# Display reconstructed images
				for j in range(n):
						# Draw the reconstructed digits
						canvas_recon[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
								g[j].reshape([28, 28])

		print("Original Images")
		plt.figure(figsize=(n, n))
		plt.imshow(canvas_orig, origin="upper", cmap="gray")
		plt.show()

		print("Reconstructed Images")
		plt.figure(figsize=(n, n))
		plt.imshow(canvas_recon, origin="upper", cmap="gray")
		plt.show()

	def encode(self, data):
		return self.sess.run([self.latent], feed_dict={self.X : data})
	def decode(self, data):
		print('in decoder')
		return self.sess.run([self.decoded], feed_dict = {self.Y: data} )
	def test(self, data):
		return self.sess.run(self.decoder_op, feed_dict={self.X:data})

if __name__ == '__main__':
	print('x')
	d = [ [1,1] , [2,2] , [3,3],  [4,4] , [5,5],  [6,6] , [7,7] , [8,8] , [9,9], [10,10], [11,11]]
	d = (np.asarray(d) / 10) - 0.3
	print(d)
	D = data_wrap(d,batch_size=11)
	a = autoencoder(D,num_step=8000)

	a.train()
	latent = np.asarray( a.encode(d) )
	latent  = np.reshape(latent, [11,-1] )
	print(a.decode(latent))
	print(a.test(d))