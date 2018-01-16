import tensorflow as tf
import numpy as np


def construct_path(path):
	def compute_path(input):
		with tf.name_scope(path['name'] + '/layer'):
			tmp = input
			for num, layer in enumerate(path['layers']):
				tmp = tf.layers.dense(tmp, layer, activation=path['activation'], name=path['name'] +'/'+str(num))
		return tmp

	return compute_path

if __name__ == '__main__':
	path = {}
	path['layers'] = [32,16,1]
	path['loss'] = 'softmax'
	path['target'] = 'sa'
	path['name'] = 'hamada'
	path['input'] = ['d1']
	path['type'] = 0
	path['feed'] = 'd1'
	path['activation']=tf.nn.sigmoid
	path['options'] ={}
	paths = [path]

	input = np.ones([50,50])
	output = np.ones([50,1])

	with tf.name_scope('inputs'):
		X = tf.placeholder(dtype='float',shape=[50,50], name='d1')
	with tf.name_scope('targets'):
		Y = tf.placeholder(dtype='float', shape=[50,1], name ='y1')

	p = construct_path(path)

	out_path1 = p(X)
	with tf.name_scope(path['name']+'/loss'):
		loss = tf.reduce_sum( tf.nn.softmax_cross_entropy_with_logits(logits= out_path1, labels=Y), name='path1_loss' )

	with tf.name_scope(path['name']+'/op'):
		train_op = tf.train.AdamOptimizer(name= 'path1_optimizer').minimize(loss)

	with tf.Session() as  sess:

		sess.run(tf.global_variables_initializer())
		for i in range(50000):
			t, l = sess.run([train_op, loss], feed_dict={X : input, Y : output})
			if (i %100 == 0):
				print(l)
				print (sess.run([out_path1], feed_dict={X: input}) )
		writer = tf.summary.FileWriter('Learning/graph2')
		writer.add_graph(sess.graph)

