import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # ERROR
import tensorflow as tf
import itertools
from sklearn import metrics, preprocessing
import numpy as np

def loss_exec(loss_fn):
		def custom(pred, labels):
			i = tf.reduce_mean(loss_fn(logits=pred, labels=labels))
			return i
		return custom

def loss_map(type, weight=1):
	if type == 'softmax':
		def fn(pred, labels):
			i = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=labels) )
			return i * weight
	if type == 'sigmoid':
		def fn(pred, labels):
			i = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=labels) )
			return i * weight
	# if type == 'sparsemax':
	# 	def fn(pred, labels):
	# 		return tf.reduce_mean( tf.contrib.sparsemax )
	return fn

def construct_path(name, layers, batch_norm=False, dropout=False, dropout_rate=0.5, noise=False, noise_std=0.1, ker_reg=None, activation=tf.nn.relu):
	def compute_path(input, phase):
		with tf.variable_scope(name + '/layers', reuse=tf.AUTO_REUSE ):
			tmp = input
			for num, layer in enumerate(layers):
				if noise:
					add = tf.random_normal(shape=tf.shape(tmp), mean=0.0, stddev=noise_std, dtype=tf.float32)
					tmp = tf.cond(phase, lambda: tf.add(tmp, add), lambda: tmp)
				if dropout:
					tmp = tf.layers.dropout(tmp, rate=dropout_rate, training=phase)
					tmp = tf.cond(phase, lambda: tmp, lambda: tf.scalar_mul(1-dropout_rate, tmp))
				tmp = tf.layers.dense(tmp, layer, activation=None, name=str(num),
															kernel_regularizer= ker_reg, reuse=tf.AUTO_REUSE)
				if batch_norm:
					tmp = tf.layers.batch_normalization(tmp, training=True)
				if activation is not None:
					tmp = activation(tmp)

		return tmp
	return compute_path

def softmax_treshold_layer(name, size):
	def fn(logits, phase):
		with tf.variable_scope(name+ '/softmax_thresholding_layer', reuse=tf.AUTO_REUSE):
			b = tf.get_variable(name='bias', shape=(size))
			tmp = tf.nn.softmax(logits)
			tmp= tf.add(tmp,b)
		return tmp
	return fn

def softmax_predictor():
	def predict(logits):
		x = tf.nn.softmax(logits)
		return tf.one_hot(tf.argmax(x, 1), logits.get_shape()[1])
	return predict

def thresholded_softmax_predictor():
	def predict(logits):
		x = tf.nn.softmax(logits)
		thresh = tf.constant(0.12, shape= logits.get_shape()[1:], dtype=tf.float32)
		return tf.greater(x, thresh )
	return predict

def sigmoid_predictor():
	def predict(logits):
		x = tf.nn.sigmoid(logits)
		return tf.round(x)
	return predict

def sparsemax_layer():
	def fn(logits, phase):
		return tf.contrib.sparsemax(logits)
	return fn

#this function takes network predictions and labels with one-hot
def f1_from_integrated_predictions(pred, labels):
	# try:
	# 	print('micro', metrics.f1_score(labels, pred, average='micro'))
	# 	print('macro', metrics.f1_score(labels, pred, average='macro'))
	# 	print('weight', metrics.f1_score(labels, pred, average='weighted'))
	# except:
	# 	print('x')
	try:
		print('samples', metrics.f1_score(labels, pred, average='micro'))
	except Exception as a:
		print(a)
		print(labels)
		print(pred)


#NOTE: If a path is fed from another path, it should appear after it in the paths list
#To create readymade paths, pass the function in path[computation]

def config_single_graph():
	paths = []

	path = {}
	path['input_dim'] = 4107
	path['name'] = 'shared1'
	path['computation'] = construct_path(path['name'], [512, 512], batch_norm=False, dropout=True, dropout_rate=0.5, noise=False, noise_std=0.16)
																			 #,ker_reg=tf.contrib.layers.l1_l2_regularizer(scale_l1=0.001, scale_l2=0.005, scope=None))
	path['input'] = 'hannah_organic'
	paths.append(path)



	path = {}
	path['name'] = 'rest_aspects'
	path['input'] = 'shared1'
	path['input_dim'] = 512
	path['computation'] = construct_path(path['name'], [12], batch_norm=False, activation=None)
																			 #,ker_reg=tf.contrib.layers.l1_l2_regularizer(scale_l1=0.0001, scale_l2=0.0005, scope=None))
	path['optimizer'] = tf.train.AdamOptimizer(name='path4_optimizer', learning_rate=0.0001 , beta1=0.85 , beta2=0.995)
	path['loss'] = loss_map('sigmoid')
	path['predictor'] = sigmoid_predictor()
	paths.append(path)

	#
	# path = {}
	# path['name'] = 'lap_aspects'
	# path['input'] = 'lap_aspects_comp'
	# path['input_dim'] = 88
	# path['computation'] = softmax_treshold_layer(path['name'], path['input_dim'])
	# path['optimizer'] = tf.train.AdamOptimizer(name='path4_optimizer')
	# path['loss'] = loss_map('sigmoid')
	# path['predictor'] = sigmoid_predictor()
	# paths.append(path)

	return paths

def config_organic_graph():
	paths = []

	path = {}
	path['input_dim'] = 4130
	path['name'] = 'shared1'
	path['computation'] = construct_path(path['name'], [512, 512], batch_norm=False, dropout=True, dropout_rate=0.5, noise=False, noise_std=0.1)
																			 #,ker_reg=tf.contrib.layers.l1_l2_regularizer(scale_l1=0.001, scale_l2=0.005, scope=None))
	path['input'] = 'hannah_organic'
	paths.append(path)


	path = {}
	path['name'] = 'entities'
	path['input'] = 'shared1'
	path['input_dim'] = 512
	path['computation'] = construct_path(path['name'], [44], batch_norm=False, activation=None)
	path['loss'] = loss_map('sigmoid')
	path['predictor'] = sigmoid_predictor()
	path['optimizer'] = tf.train.AdamOptimizer(name='entities_optimizer', learning_rate=0.0001, beta1=0.85, beta2=0.995)
	paths.append(path)

	# path = {}
	# path['name'] = 'attrs'
	# path['input'] = 'shared1'
	# path['input_dim'] = 512
	# path['computation'] = construct_path(path['name'], [6], batch_norm=False, activation=tf.nn.relu)
	# path['loss'] = loss_map('softmax')
	# path['predictor'] = softmax_predictor()
	# path['optimizer'] = tf.train.AdamOptimizer(name='attrs_optimizer', learning_rate=0.0001, beta1=0.85, beta2=0.995)
	# paths.append(path)


	# path = {}
	# path['name'] = 'aspect_comp'
	# path['input'] = 'shared1'
	# path['input_dim'] = 512
	# path['computation'] = construct_path(path['name'], [44], batch_norm=False, activation=tf.nn.relu)
	# 																		 #,ker_reg=tf.contrib.layers.l1_l2_regularizer(scale_l1=0.0001, scale_l2=0.0005, scope=None))
	# # path['optimizer'] = tf.train.AdamOptimizer(name='path4_optimizer', learning_rate=0.0001 , beta1=0.85 , beta2=0.995)
	# # path['loss'] = loss_map('sigmoid')
	# # path['predictor'] = sigmoid_predictor()
	# paths.append(path)
	#
	#
	# path = {}
	# path['name'] = 'aspects'
	# path['input'] = 'aspect_comp'
	# path['input_dim'] = 44
	# path['computation'] = softmax_treshold_layer(path['name'], path['input_dim'])
	# path['optimizer'] = tf.train.AdamOptimizer(name='path4_optimizer', learning_rate=0.0001 , beta1=0.85 , beta2=0.995)
	# path['loss'] = loss_map('sigmoid')
	# path['predictor'] = sigmoid_predictor()
	# paths.append(path)

	return paths


def config_sentiment_graph():
	paths = []

	path = {}
	path['input_dim'] = 4096
	path['name'] = 'shared1'
	path['computation'] = construct_path(path['name'], [512, 512], batch_norm=False, dropout=True, dropout_rate=0.5, noise=False, noise_std=0.1)
																			 #,ker_reg=tf.contrib.layers.l1_l2_regularizer(scale_l1=0.001, scale_l2=0.005, scope=None))
	path['input'] = 'sent'
	paths.append(path)


	path = {}
	path['name'] = 'sent'
	path['input'] = 'shared1'
	path['input_dim'] = 512
	path['computation'] = construct_path(path['name'], [3], batch_norm=False, activation=None)
	path['loss'] = loss_map('softmax')
	path['predictor'] = softmax_predictor()
	path['optimizer'] = tf.train.AdamOptimizer(name='sent_optimizer', learning_rate=0.0001)
	paths.append(path)

	return paths

def config_joint_graph():
	paths = []

	path = {}
	path['input_dim'] = 4096
	path['name'] = 'shared1'
	path['computation'] = construct_path(path['name'], [512, 256], batch_norm=False, dropout=True, dropout_rate=0.5, noise=False, noise_std=0.16)
																			 #,ker_reg=tf.contrib.layers.l1_l2_regularizer(scale_l1=0.001, scale_l2=0.005, scope=None))
	path['input'] = 'hannah_organic'
	paths.append(path)


	path = {}
	path['name'] = 'lap_aspects'
	path['input'] = 'shared1'
	path['input_dim'] = 512
	path['computation'] = construct_path(path['name'], [88], batch_norm=False, activation=None)
																			 #,ker_reg=tf.contrib.layers.l1_l2_regularizer(scale_l1=0.0001, scale_l2=0.0005, scope=None))
	path['optimizer'] = tf.train.AdamOptimizer(name='path1_optimizer', learning_rate=0.0001 , beta1=0.85 , beta2=0.995)
	path['loss'] = loss_map('sigmoid')
	path['predictor'] = sigmoid_predictor()
	paths.append(path)
	#
	path = {}
	path['name'] = 'rest_aspects'
	path['input'] = 'shared1'
	path['input_dim'] = 512
	path['computation'] = construct_path(path['name'], [12], batch_norm=False, activation=None)
																			 #,ker_reg=tf.contrib.layers.l1_l2_regularizer(scale_l1=0.0001, scale_l2=0.0005, scope=None))
	path['optimizer'] = tf.train.AdamOptimizer(name='path2_optimizer', learning_rate=0.0001 , beta1=0.85 , beta2=0.995)
	path['loss'] = loss_map('sigmoid')
	path['predictor'] = sigmoid_predictor()
	paths.append(path)

	path = {}
	path['name'] = 'org_aspects'
	path['input'] = 'shared1'
	path['input_dim'] = 512
	path['computation'] = construct_path(path['name'], [77], batch_norm=False, activation=None)
																			 #,ker_reg=tf.contrib.layers.l1_l2_regularizer(scale_l1=0.0001, scale_l2=0.0005, scope=None))
	path['optimizer'] = tf.train.AdamOptimizer(name='path2_optimizer', learning_rate=0.0001 , beta1=0.85 , beta2=0.995)
	path['loss'] = loss_map('sigmoid')
	path['predictor'] = sigmoid_predictor()
	paths.append(path)


	return paths


def config_graph(c):
	paths = []

	path = {}
	path['name'] = 'shared1'
	path['input_dim'] = 4096
	x = c[path['name']+'/layers'][-1]
	print(x)
	path['output_dim'] = x
	path['computation'] = construct_path( path['name'], c[path['name']+'/layers'], batch_norm=c['bn'], dropout=c['dp'],
																				noise=c['noise'], ker_reg=c['kr'], activation=c['actv'] )
	path['input'] = 'semeval'
	paths.append(path)

	# path = {}
	# path['name'] = 'sentiment'
	# path['input'] = 'shared1'
	# path['input_dim'] = 2048
	# path['layers'] = [512,256,4]
	# path['optimizer'] = tf.train.AdamOptimizer(name='path1_optimizer')
	# path['activation'] = tf.nn.relu
	# path['loss'] = 'softmax'
	# path['reuse'] = True
	# path['options'] = ('batch_norm')
	# paths.append(path)
	path = {}
	path['name'] = 'entities'
	path['input'] = 'shared1'
	path['input_dim'] = paths[0]['output_dim']
	path['computation'] = construct_path( path['name'], c[path['name']+'/layers'] , batch_norm=c['bn'], dropout=c['dp'],
																				noise=c['noise'], ker_reg=c['kr'], activation=c['actv'] )
	path['optimizer'] = tf.train.AdamOptimizer(name='path2_optimizer' )
	path['loss'] = loss_map('softmax')
	path['predictor'] = softmax_predictor()
	paths.append(path)

	path = {}
	path['name'] = 'attributes'
	path['input'] = 'shared1'
	path['input_dim'] = paths[0]['output_dim']
	path['computation'] = construct_path( path['name'], c[path['name']+'/layers'] , batch_norm=c['bn'], dropout=c['dp'],
																				noise=c['noise'], ker_reg=c['kr'], activation=c['actv'] )
	path['optimizer'] = tf.train.AdamOptimizer(name='path3_optimizer')
	path['loss'] = loss_map('softmax')
	path['predictor'] = softmax_predictor()

	paths.append(path)

	return paths


def graph_config_generator():
	config = {}
	config['shared1/layers'] = [[1024],[1024, 512]]
	config['entities/layers'] = [[128, 7],[512, 7]]
	config['attributes/layers'] =[[128, 6], [512, 6]]

	config['bn'] = [True, False]
	config['dp'] = [True, False]
	config['noise'] = [False]
	config['actv'] = [tf.nn.relu]
	config['kr'] = [tf.contrib.layers.l1_l2_regularizer(scale_l1=0.5, scale_l2=0.5, scope=None),
									tf.contrib.layers.l1_l2_regularizer(scale_l1=0.01, scale_l2=1.0, scope=None),
									tf.contrib.layers.l1_l2_regularizer(scale_l1=1.0, scale_l2=0.1, scope=None),
									None]

	combinations = itertools.product(config['shared1/layers'], config['entities/layers'], config['attributes/layers'],
																	 config['bn'], config['dp'], config['noise'], config['actv'], config['kr'] )
	configurations = []
	for c in combinations:
		configuration = {}
		print(c, '\n---------------')
		configuration['shared1/layers'] = c[0]
		configuration['entities/layers'] = c[1]
		configuration['attributes/layers'] = c[2]
		configuration['bn'] = c[3]
		configuration['dp'] = c[4]
		configuration['noise'] = c[5]
		configuration['actv'] = c[6]
		configuration['kr'] = c[7]

		configurations.append(configuration)
		print(configuration, '\n-------------------------------\n')

	print(len(configurations))
	return configurations

if __name__ == '__main__':
	graph_config_generator()