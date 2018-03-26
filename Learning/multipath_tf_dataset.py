import tensorflow as tf
import numpy as np
import os
from datetime import datetime
from Learning.config_generator import *


def dataset_generator(dataset, holdout=None):
	length = len(dataset['features'])
	print('dataset length ', length)
	if holdout is not None:
		p = np.random.randint(0, length, size=holdout)
		def train_gen():
			for i in range (length):
				if(i in p):
					# print(i)
					continue
				labels = [ x['features'][i] for x in  dataset['tasks'] ]
				if(isinstance(dataset['features'][i], str)):
					dataset['features'][i] = dataset['features'][i].encode('utf-8')
				labels.insert(0, dataset['features'][i])
				yield tuple(labels)
		def cv_gen():
			for i in p:
				labels = [ x['features'][i] for x in  dataset['tasks'] ]
				if(isinstance(dataset['features'][i], str)):
					dataset['features'][i] = dataset['features'][i].encode('utf-8')
				labels.insert(0, dataset['features'][i])
				yield tuple(labels)
		return train_gen, cv_gen
	else:
		def gen():
			for i in range(length):
				labels = [x['features'][i] for x in dataset['tasks']]
				if (isinstance(dataset['features'][i], str)):
					dataset['features'][i] = dataset['features'][i].encode('utf-8')
				labels.insert(0, dataset['features'][i])
				yield tuple(labels)

	return gen

def inference_generator(data):
	def gen():
		for i in data:
			yield i
	return gen

def build_tf_dataset(datasets, sess):
	handles = {}
	for dataset in datasets:
		with tf.name_scope('Datasets/' + dataset['name']):
			task_iterators = {}
			dataset_handles = {}
			if 'batch_size' not in dataset.keys():
				batch_size = 100
			else:
				print('batch size: ',  dataset['batch_size'])
				batch_size = dataset['batch_size']

			dataset_col_labels = ['data']
			types = [dataset['type']]
			for task in dataset['tasks']:
				types.append(task['type'])
				dataset_col_labels.append(task['name'])

			tf.set_random_seed(np.random.randint(0,10000))

			if 'holdout' in dataset:
				h = dataset['holdout']
				gen, cv_gen = dataset_generator(dataset, holdout = h)
				d = tf.data.Dataset.from_generator(cv_gen, output_types=tuple(types))
				d = d.repeat().batch(h)
				# d = d.prefetch(h * 3)

				with tf.name_scope('features_cv'):
					d_feat = d.map(lambda *x: x[0]).prefetch(h)
					d_iter = d_feat.make_initializable_iterator()
					d_iter.make_initializer(d_feat)
					sess.run(d_iter.initializer)
					dataset_handles['data_cv'] = sess.run(d_iter.string_handle())

				for i in range(1, len(dataset_col_labels)):
					with tf.name_scope(dataset_col_labels[i] + '_cv'):
						data = (d.map(lambda *x: x[i])).prefetch(h)
						data_iter = data.make_initializable_iterator()
						data_iter.make_initializer(data)
						sess.run(data_iter.initializer)
						task_iterators[dataset_col_labels[i]] = data_iter
						dataset_handles[dataset_col_labels[i]+'_cv'] = sess.run(task_iterators[dataset_col_labels[i]].string_handle())
		
			else:
				gen = dataset_generator(dataset)

			d = tf.data.Dataset.from_generator(gen, output_types=tuple(types))
			# d = d.shuffle(buffer_size=1200)
			d = d.repeat().batch(batch_size)
			#d = d.prefetch(1500)

			with tf.name_scope('features'):
				d_feat = d.map(lambda *x:  x[0], num_parallel_calls=8).prefetch(batch_size)
				d_iter = d_feat.make_initializable_iterator()
				d_iter.make_initializer(d_feat)
				sess.run(d_iter.initializer)
				dataset_handles['data'] = sess.run(d_iter.string_handle())

			for i in range(1,len(dataset_col_labels)):
				with tf.name_scope(dataset_col_labels[i]):
					data = d.map(lambda *x:  x[i],  num_parallel_calls=8).prefetch(batch_size)
					data_iter = data.make_initializable_iterator( )
					data_iter.make_initializer(data)
					sess.run(data_iter.initializer)
					task_iterators[dataset_col_labels[i]] = data_iter
					dataset_handles[dataset_col_labels[i]] = sess.run( task_iterators[dataset_col_labels[i]].string_handle() )

			handles[dataset['name']] = dataset_handles
	print(handles, '\n-------------------------------')
	return handles


#This is a constructor for a multi_task, multi_data tf classfier
#paths are computational paths ending with a loss function, or feeding other paths.
class TfMultiPathClassifier():
	def __init__(self, datasets, paths, params={}):

		self.mydir = os.path.join(os.getcwd(), 'logs', datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

		tf.reset_default_graph()
		self.sess = tf.Session()
		self.phase = tf.placeholder(tf.bool, name='phase')
		self.cv = tf.placeholder(tf.bool, name='crossvalidating')
		self.params= params

		self.accuracy= {}
		self.feed = {}
		self.inputs = {}
		self.targets = {}
		self.comp = {}
		self.logits = {}
		self.loss = {}
		self.opt = {}
		self.predictors={}
		self.iterators = {}
		self.handles = build_tf_dataset(datasets, self.sess)
		self.handle_placeholders = {}
		self.summaries = []
		self.cv_summaries= []
		self.dataset_paths={}

		#prep datasets
		self.dataset_tasks={}
		for dataset in datasets:
			tasks=[]
			for task in dataset['tasks']:
				tasks.append(task['name'])
			self.dataset_tasks[dataset['name']] = tasks

		self.build_graph(paths)


	def build_graph(self, paths):
		for p in paths:
			id = p['name']
			with tf.variable_scope(id + '/feed', reuse=True):
				if (p['input'] not in self.comp.keys()):
					self.handle_placeholders[id + 'input'] = tf.placeholder(tf.string, shape=[])
					if('input_string' in p):
						self.iterators[id + 'input'] = tf.data.Iterator.from_string_handle(
							self.handle_placeholders[id + 'input'], tf.string, output_shapes=[None, p['input_dim'] ])  # TODO change input_types to be configurable
					else:
						self.iterators[id + 'input'] = tf.data.Iterator.from_string_handle(
							self.handle_placeholders[id + 'input'] ,tf.float32, output_shapes=[None, p['input_dim']] )	#TODO change input_types to be configurable
					self.inputs[id] = self.iterators[id + 'input'].get_next()
					self.feed[id] = id
				else:
					self.inputs[id] = self.logits[p['input']]
					self.feed[id] = self.feed[p['input']]
				if('loss' in p):
					self.handle_placeholders[id + 'target'] = tf.placeholder(tf.string, shape=[])
					self.iterators[id + 'target'] = tf.data.Iterator.from_string_handle(
						self.handle_placeholders[id + 'target'], tf.float32)  # TODO change input_types to be configurable
					self.targets[id] = self.iterators[id + 'target'].get_next()

			with tf.variable_scope('', reuse=True):
				self.comp[id] = p['computation']
				self.logits[id] = self.comp[id](self.inputs[id], self.phase)

			if ('predictor' in p):
				with tf.variable_scope(id + 'predictor', reuse=True):
					self.predictors[id] = p['predictor'](self.logits[id])
			else:
				with tf.name_scope(id + 'predictor'):
					self.predictors[id] = self.logits[id]

			if ('loss' in p):
				with tf.variable_scope(id + '/loss', reuse=True):
					self.loss[id] = p['loss'](self.logits[id], self.targets[id])
					self.summaries.append(tf.summary.scalar(id + ' loss', self.loss[id]))
					self.cv_summaries.append(tf.summary.scalar(id + ' loss_cv', self.loss[id]))

				with tf.variable_scope(id + '/optimizer', reuse=tf.AUTO_REUSE):
					self.opt[id] = (p['optimizer'].minimize(self.loss[id]))


		for i in self.dataset_tasks:
			self.dataset_paths[i] = {}
			self.dataset_paths[i]['losses'] = [self.loss[x] for x in self.opt.keys() if x in self.dataset_tasks[i]]
			self.dataset_paths[i]['optimizers'] = [self.opt[x] for x in self.loss.keys() if x in self.dataset_tasks[i]]
			self.dataset_paths[i]['feeds'] = [self.feed[x] for x in self.feed.keys() if x in self.dataset_tasks[i]]
			self.dataset_paths[i]['predictors'] = [self.predictors[x] for x in sorted(self.predictors.keys()) if x in self.dataset_tasks[i]]
			self.dataset_paths[i]['targets'] = [self.targets[x] for x in sorted(self.targets.keys()) if x in self.dataset_tasks[i]]

			input_feed= [(self.handle_placeholders[x + 'input'], self.handles[i]['data']) for x in self.dataset_paths[i]['feeds']]
			target_feed=[(self.handle_placeholders[x+'target'], self.handles[i][x]) \
									 for x in self.dataset_tasks[i] if x+ 'target' in self.handle_placeholders]
			self.dataset_paths[i]['feed_dict'] = dict(input_feed + target_feed)
			self.dataset_paths[i]['feed_dict'][self.phase] = True
			if 'data_cv' in self.handles[i]:
				input_feed = [(self.handle_placeholders[x + 'input'], self.handles[i]['data_cv']) for x in
											self.dataset_paths[i]['feeds']]
				target_feed = [(self.handle_placeholders[x + 'target'], self.handles[i][x+'_cv']) \
											 for x in self.dataset_tasks[i] if x + 'target' in self.handle_placeholders]
				self.dataset_paths[i]['feed_dict_cv'] = dict(input_feed + target_feed)
				self.dataset_paths[i]['feed_dict_cv'][self.phase] = False

		self.merged = tf.summary.merge(self.summaries, 'train summaries')
		self.merged_cv = tf.summary.merge(self.cv_summaries, 'cv summaries')
		self.train_writer = tf.summary.FileWriter(os.path.join(self.mydir , 'summaries'), self.sess.graph)


	def train(self):
		self.sess.run(tf.global_variables_initializer())
		self.sess.run(tf.local_variables_initializer())

		for i in range(self.params['train_iter']):
			cv_counter = 0
			for dataset in self.dataset_paths.values():
				# _, loss, summaries, acc = self.sess.run(
				# 	[dataset['optimizers'], dataset['losses'], self.merged, self.accuracy],\
				# 	feed_dict= dataset['feed_dict'] )
				_, loss= self.sess.run(
					[dataset['optimizers'], dataset['losses']],\
					feed_dict= dataset['feed_dict'] )
				if(i % 500 == 0):
					#self.train_writer.add_summary(summaries, cv_counter)
					# print(acc)
					print(loss, i, '\n-------\n')
					if('feed_dict_cv' in dataset):
						print('cross validating')
						# loss, summaries, pred, ll = self.sess.run\
						# 	([dataset['losses'], self.merged_cv, dataset['predictors'], dataset['targets']],
						# 	 feed_dict= dataset['feed_dict_cv'])
						loss, pred, ll = self.sess.run\
							([dataset['losses'], dataset['predictors'], dataset['targets']],
							 feed_dict= dataset['feed_dict_cv'])

						for out_counter, p in enumerate(pred):
							f1_from_integrated_predictions(pred=p, labels=ll[out_counter])

						#self.train_writer.add_summary(summaries, cv_counter)
						print('cv loss', loss, '\n----------------------\n')
						cv_counter += 1

	def save(self):
		writer = tf.summary.FileWriter(os.path.join(self.mydir, 'graph'))
		writer.add_graph(self.sess.graph)

	def get_raw_prediciton(self, pathname, input):
		gen = inference_generator(input)
		d = tf.data.Dataset.from_generator(gen, output_types= tf.float32)
		x = len(input)
		d = d.batch(x)
		iter = d.make_one_shot_iterator()
		handle = self.sess.run(iter.string_handle())
		if(pathname in self.logits):
			return self.logits[pathname].eval(session= self.sess, feed_dict =
			{self.phase:False, self.handle_placeholders[self.feed[pathname]+'input' ]: handle })

	def get_prediciton(self, pathname, input):
		gen = inference_generator(input)
		d = tf.data.Dataset.from_generator(gen, output_types=tf.float32)
		x = len(input)
		d = d.batch(x)
		iter = d.make_one_shot_iterator()
		handle = self.sess.run(iter.string_handle())
		if (pathname in self.predictors):
			return self.predictors[pathname].eval(session=self.sess, feed_dict=
			{self.phase: False, self.handle_placeholders[self.feed[pathname] + 'input']: handle})

if __name__ == '__main__':
	inputs={}
	inputs['d1'] = np.zeros([2,2])
	inputs['d2'] = np.ones([3,3])
	outputs={}



	paths = config_graph()

	datasets = []
	dataset={}
	dataset['name'] = 'd1'
	dataset['features'] = np.ones([500 , 5])
	dataset['targets'] = np.zeros([500,5])
	dataset['type'] = tf.float32
	for i in range(500):
		dataset['features'][i] *= i
		dataset['targets'][i][0] += 1

	dataset['tasks'] = [{'name' : 'semeval_sub1', 'features' : dataset['features']/5 , 'type' : tf.float32 },\
											{'name' : 'semeval_sub2', 'features' : dataset['features']*5, 'type' : tf.float32   }, \
											{'name': 'semeval_sub3', 'features': 	dataset['targets'],  'type' : tf.float32 }]


	datasets = [dataset]

	#itera,_=build_tf_dataset(datasets)


	# for i in range(10):
	# 	print( sess.run(itera['d1'].get_next() ))
	M = TfMultiPathClassifier(datasets, inputs, outputs, paths)
	M.save()
	M.train()
	print("resultssss   ",M.get_logits([[1,1,1,7,1,5,1,0,1,1]], 'semeval_sub3') )
