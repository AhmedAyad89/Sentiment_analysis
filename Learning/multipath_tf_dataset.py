import tensorflow as tf
import numpy as np

def loss_map(loss):
	if (loss == 'softmax'):
		def cross_entropy(pred, labels):
			i = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=labels) )
			return i
		return cross_entropy

def construct_path(path):
		def compute_path(input):
			with tf.variable_scope(path['name'] + '/layers', reuse=False):
				tmp = input
				for num, layer in enumerate(path['layers']):
					tmp = tf.layers.dense(tmp, layer, activation=path['activation'], name=str(num))
			return tmp
		return compute_path

def config_graph():
	paths = []

	path ={}
	path['input_dim'] = 50
	path['layers'] = [50,10]
	path['target'] ='sa'
	path['activation'] = tf.nn.relu
	path['name'] = 'shared1'
	path['input'] = 'd1'
	path['feed'] = 'd1'
	path['reuse'] = True
	paths.append(path)

	path ={}
	path['optimizer'] = tf.train.AdamOptimizer(name= 'path1_optimizer')
	path['activation'] = tf.nn.relu
	path['layers'] = [10,5]
	path['loss'] = 'softmax'
	path['target'] ='sa'
	path['name'] = 'semeval_sub1'
	path['input'] = 'shared1'
	path['feed'] = 'd1'
	path['reuse'] = True
	path['input_dim'] = 10
	paths.append(path)

	return paths

def dataset_generator(dataset):
	label_text= ['data', 'l1', 'l2', 'l3']
	def gen():
		for i in range (len(dataset['features']) ):
			labels = [ x['features'][i] for x in  dataset['tasks'] ]
			labels.insert(0,dataset['features'][i].encode('utf-8'))
			x = [dataset['features'][i], labels ]
			yield tuple(labels)
	return gen

def build_tf_dataset(datasets):
	print('x')

	sess = tf.Session()
	d = []
	handles=[]
	for dataset in datasets:


		gen = dataset_generator(dataset)
		g = gen()
		x = next(g)
		print(x[0])
		types = []
		types.append(dataset['type'])
		for task in dataset['tasks']:
			types.append(task['type'])
		print(types)
		d = tf.data.Dataset.from_generator(gen, output_types=tuple(types))
		#d = d.shuffle(buffer_size=32)
		#d = d.repeat(2)

		dataset_iterator = d.make_one_shot_iterator()
		d_handle = dataset_iterator.string_handle()
		handles.append(d_handle)


		it = d.make_one_shot_iterator()
		for i in range(2):
			t = sess.run(it.get_next())
			#print(t[0], '/', t[1],'/', t[2])
			print('\n++++++++++')
	handle = tf.placeholder(tf.string, shape=[])
	iterator = tf.data.Iterator.from_string_handle(handle, dataset_iterator.output_types)

	return iterator, handles, handle, sess



#This is a constructor for a multi_task, multi_data tf classfier
#inputs contains a dict of inputs, each input is list of data features. Dict keys are "datasetname"
#outputs is a dict of outputs. each value is a list of labels. Dict keys are "datasetname#taskname"
#paths are computational paths ending with a loss function
class tf_multi_path_classifer():
	def __init__(self, datasets, inputs, outputs, paths):
		tf.reset_default_graph()

		#prep datasets
		self.dataset_tasks=[]
		for dataset in datasets:
			tasks=[]
			for task in dataset['tasks']:
				tasks.append(task['name'])
			self.dataset_tasks.append(tasks)
		print(tasks)

		self.sess = tf.Session()
		self.feed = {}
		self.inputs = {}
		self.targets = {}
		self.comp={}
		self.logits={}
		self.loss={}
		self.opt ={}

		for p in paths:
			with tf.name_scope(p['name']+ '/feed' ):
				if (p['input'] not in self.comp.keys()):
					self.inputs[p['name']] = tf.placeholder("float", [None, p['input_dim']], name = 'features')
					self.feed[p['name']] = self.inputs[p['name']]
					if ('loss' in p):
						self.targets[p['name']] = tf.placeholder("float", [None, p['layers'][-1] ], name = 'targets')
				else:
					self.inputs[p['name']] = self.logits[p['input']]
					self.feed[p['name']] = self.feed[p['input']]
					if('loss' in p):
						self.targets[p['name']] = tf.placeholder("float", [None, p['layers'][-1]], name='targets')

			with tf.name_scope(''):
				self.comp[p['name']] = construct_path(p)
				self.logits[p['name']] = self.comp[p['name']](self.inputs[p['name']])

			if ('loss' in p):
				with tf.name_scope(p['name'] + '/loss'):
					self.loss[p['name']] = loss_map(p['loss'])(self.logits[p['name']], self.targets[p['name']] )
				with tf.name_scope(p['name'] + '/optimizer'):
					self.opt[p['name']] = p['optimizer'].minimize(self.loss[p['name']])

		self.datasets_optimizers=[]
		self.datasets_losses = []
		self.datasets_feeds = []
		for i in self.dataset_tasks:
			self.datasets_losses.append			([self.loss[x] for x in self.opt.keys() if x in i  ] )
			self.datasets_optimizers.append	([self.opt[x] for x in self.loss.keys() if x in i  ] )
			self.datasets_feeds.append			([self.feed[x] for x in self.feed.keys() if x in i ])
		print('dataset losses ' , self.datasets_losses)
		print('dataset optimizers ', self.datasets_optimizers )

	def train(self):
		self.sess.run(tf.global_variables_initializer())
		for c, dataset in enumerate (datasets):
			for i in range(100):
				# t, l = self.sess.run([ self.datasets_optimizers[c], self.datasets_losses[c] ] , \
				# 								feed_dict= {self.feed[p['name']] : dataset['features'],\
				# 														self.targets[p['name']] : task['features'] })
				# print(t,l)
				print()
				if(i % 100 ==0):
					print( self.sess.run([self.logits['semeval_sub1']], feed_dict={\
						self.feed['semeval_sub1'] : datasets[0]['features']} ) )
					# print(self.sess.run(self.get_logits(datasets[0]['features']) ))

	def save(self):
		writer = tf.summary.FileWriter('graph/graph')
		writer.add_graph(self.sess.graph)

	def get_logits(self, input):
		input = tf.convert_to_tensor(input, tf.float32)
		return self.comp['semeval_sub1'](input)

if __name__ == '__main__':
	print(tf.__version__)
	inputs={}
	inputs['d1'] = np.zeros([2,2])
	inputs['d2'] = np.ones([3,3])
	outputs={}
	outputs


	paths = config_graph()

	datasets = []
	dataset={}
	dataset['name'] = 'd1'
	dataset['features'] = np.ones([50 , 50])
	dataset2 = np.ones([50 , 50])
	for i in range(50):
		dataset['features'][i] *= i
		dataset2[i] *= i+100

	dataset['tasks'] = [{'name' : 'semeval_sub1', 'features' : dataset['features']/5  },\
											{'name' : 'semeval_sub2', 'features' : dataset['features']*5  }, \
											{'name': 'semeval_sub3', 'features': np.ones([50,1])}]
	datasets = [dataset]

	gen = dataset_generator(dataset)

	# g =gen()
	# for i in g:
	# 	print(i)
	# 	print(len(i))

	build_tf_dataset(datasets)
	# with tf.Session() as sess:
	#
	# 	it, d1, d2, handle=build_tf_dataset(dataset['features'], dataset2,None)
	#
	# 	d1 = sess.run(d1)
	# 	d2 = sess.run(d2)
	# 	print(d1)
	# 	while(True):
	# 		print( sess.run(it.get_next(), feed_dict={handle : d1 } ) )
	# 		print(sess.run(it.get_next(), feed_dict={handle: d2}))
	#
	# 		print('\n-----------------------')


	#
	#
	# M = tf_multi_path_classifer(datasets,inputs,outputs, paths)
	# M.save()
	# M.train()
