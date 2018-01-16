from __future__ import print_function
from random import shuffle

import tensorflow as tf
import numpy as np


def gaussian_additive_noise_layer(input_layer, std):
	noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
	return input_layer + noise


def NN(data, labels, vectors, test_data, test_labels):
	# Network Parameters
	n_hidden_1 = 2048 # 1st layer number of neurons
	n_hidden_2 = 1024 # 2nd layer number of neurons
	num_input = 4096 # MNIST data input (img shape: 28*28)
	num_classes = 12 #len(labels[0])
	vec_size = len(vectors[0])
	print("vec size", len(vectors[0]))

	# tf Graph input
	X = tf.placeholder("float", [None, num_input])
	Y = tf.placeholder("float", [None, num_classes])

	# Store layers weight & bias
	weights = {
			'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
			'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
			'out': tf.Variable(tf.random_normal([n_hidden_2, vec_size]))
	}
	biases = {
			'b1': tf.Variable(tf.random_normal([n_hidden_1])),
			'b2': tf.Variable(tf.random_normal([n_hidden_2])),
			'out': tf.Variable(tf.random_normal([vec_size]))
	}


	# Create model
	def neural_net(x, noise = True):
			if(noise):
				x = gaussian_noise_layer(x, 0.4)
			# Hidden fully connected layer with 256 neurons
			layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
			if(noise):
				layer_1 = gaussian_noise_layer(layer_1, 0.4)
			# Hidden fully connected layer with 256 neurons
			layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])

			# Output fully connected layer with a neuron for each class
			out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
			return out_layer

	# Construct model
	logits = neural_net(X)
	test_logits = neural_net(X, False)
	prediction = tf.nn.softmax(logits)
	test_prediction = tf.nn.softmax(full_vector_loss(test_logits,vectors) )
	full_cosines = full_vector_loss(test_logits,vectors)
	cosines= vector_loss(logits, Y,vectors)
	# Define loss and optimizer
	# loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
	# 		logits=logits, labels=Y))
	loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
		 		logits=cosines, labels=tf.ones(batch_size)))
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
	train_op = optimizer.minimize(loss_op)

	# Evaluate model
	correct_pred = tf.equal(tf.argmax(test_prediction, 1), tf.argmax(Y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

	# Initialize the variables (i.e. assign their default value)
	init = tf.global_variables_initializer()


	# Start training
	with tf.Session() as sess:

			# Run the initializer
			sess.run(init)

			for step in range(1, num_steps+1):
					#batch_x, batch_y = mnist.train.next_batch(batch_size)
					batch_x, batch_y = next_batch(data, labels,step,batch_size)
					# Run optimization op (backprop)
					sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
					if step % display_step == 0 or step == 1:
							# Calculate batch loss and accuracy
							loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
																																	 Y: batch_y})
							print("Step " + str(step) + ", Minibatch Loss= " + \
										"{:.4f}".format(loss) + ", Training Accuracy= " + \
										"{:.3f}".format(acc))

			print("Optimization Finished!")

			# Calculate accuracy for MNIST test images
			print("Testing Accuracy:", \
					sess.run(accuracy, feed_dict={X: test_data,
																				Y: test_labels}))
			return sess.run(tf.argmax(test_prediction,1), feed_dict={X: test_data, Y: test_labels} )