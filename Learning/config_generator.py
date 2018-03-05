import tensorflow as tf
import itertools
def loss_exec(loss_fn):
		def custom(pred, labels):
			i = tf.reduce_mean(loss_fn(logits=pred, labels=labels))
			return i
		return custom

def loss_map(type):
	if type == 'softmax':
		def cross_entropy(pred, labels):
			i = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=labels) )
			return i
	return cross_entropy

def construct_path(name, layers, batch_norm=False, dropout=False, noise=False, ker_reg=None, activation=tf.nn.relu):
	def compute_path(input, phase):
		with tf.variable_scope(name + '/layers', reuse=tf.AUTO_REUSE ):
			tmp = input
			for num, layer in enumerate(layers):
				if dropout:
					tmp = tf.layers.dropout(tmp, rate=0.6, training=phase)
				if noise:
					add = tf.random_normal(shape=tf.shape(tmp), mean=0.0, stddev=0.01, dtype=tf.float32)
					tmp = tf.cond(phase, tmp, tf.add(tmp, add))

				tmp = tf.layers.dense(tmp, layer, activation=None, name=str(num),
															kernel_regularizer= ker_reg)
				if(batch_norm):
					tmp = tf.layers.batch_normalization(tmp, training=phase)

				tmp = activation(tmp)

		return tmp
	return compute_path

def softmax_predictor():
	def predict(logits):
		x = tf.nn.softmax(logits)
		return tf.argmax(x,1)
	return predict

#NOTE: If a path is fed from another path, it should appear after it in the paths list
#To create readymade paths, pass the function in path[computation]
#to create readymade losses, pass the loss function into path[loss_computation]
# def config_graph():
# 	paths = []
#
# 	path = {}
# 	path['input_dim'] = 4096
# 	path['name'] = 'shared1'
# 	path['computation'] = construct_path( path['name'], [2048, 1024,512], batch_norm=True)
# 	path['input'] = 'semeval'
# 	paths.append(path)
#
# 	# path = {}
# 	# path['name'] = 'sentiment'
# 	# path['input'] = 'shared1'
# 	# path['input_dim'] = 2048
# 	# path['layers'] = [512,256,4]
# 	# path['optimizer'] = tf.train.AdamOptimizer(name='path1_optimizer')
# 	# path['activation'] = tf.nn.relu
# 	# path['loss'] = 'softmax'
# 	# path['reuse'] = True
# 	# path['options'] = ('batch_norm')
# 	# paths.append(path)
#
# 	path = {}
# 	path['name'] = 'entities'
# 	path['input'] = 'shared1'
# 	path['input_dim'] = 512
# 	path['computation'] = construct_path(path['name'], [128, 7], batch_norm=True)
# 	path['optimizer'] = tf.train.AdamOptimizer(name='path2_optimizer' )
# 	path['loss'] = loss_map('softmax')
# 	#path['loss_computation'] = tf.nn.sigmoid_cross_entropy_with_logits
# 	path['predictor'] = softmax_predictor()
# 	paths.append(path)
#
# 	path = {}
# 	path['name'] = 'attributes'
# 	path['input'] = 'shared1'
# 	path['input_dim'] = 512
# 	path['computation'] = construct_path(path['name'], [128, 6], batch_norm=True)
# 	path['optimizer'] = tf.train.AdamOptimizer(name='path3_optimizer')
# 	path['loss'] = loss_map('softmax')
# 	#path['loss_computation'] = tf.nn.sigmoid_cross_entropy_with_logits
# 	path['predictor'] = softmax_predictor()
#
# 	paths.append(path)
#
# 	return paths

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
	config['shared1/layers'] = [[2048, 1024,512],[1024, 512]]
	config['entities/layers'] = [[128, 7],[512, 7]]
	config['attributes/layers'] =[[126, 6], [512, 6]]

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