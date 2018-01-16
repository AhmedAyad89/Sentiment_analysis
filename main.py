from DataSet.semEval import SemEval
from DataSet import preprocess
from FeatureExtrac.autoencoder import *
from FeatureExtrac.openAI_transfrom import openAI_transform
from Learning.multipath_tf_dataset import *
from visual import *
from utils import  *
from sklearn_pipeline import *
import pickle
import tensorflow as tf

data, labels = preprocess.flatten_single_label('DataSet/semEval/semevalRestaurant-rawTextData',\
																							'DataSet/semEval/semevalRestaurant-rawTextLabels',)

aspect_labels, aspect_labeling = preprocess.encode_labels(pickle_entities='DataSet/semEval/semevalRestaurant-Entities',\
								pickle_attrs='DataSet/semEval/semevalRestaurant-Attributes', labels=labels, scheme=[True, True, False], one_hot=True)

sentiment_labels, sentiment_labeling = preprocess.encode_labels(pickle_entities='DataSet/semEval/semevalRestaurant-Entities',\
								pickle_attrs='DataSet/semEval/semevalRestaurant-Attributes', labels=labels, scheme=[False, False, True], one_hot=True)

print(aspect_labeling)
datasets=[]
dataset = {}
dataset['name'] = 'semeval_restaurants'
dataset['type'] = tf.string
dataset['features'] = data
dataset['tasks'] = [{'name' : 'aspects', 'features' : aspect_labels, 'type': tf.float32},\
										{'name': 'sentiment', 'features': sentiment_labels, 'type': tf.float32 }]

datasets.append(dataset)

data, labels = preprocess.flatten_single_label('DataSet/semEval/semevalLaptop-rawTextData',\
																							'DataSet/semEval/semevalLaptop-rawTextLabels',)

aspect_labels, aspect_labeling = preprocess.encode_labels(pickle_entities='DataSet/semEval/semevalLaptop-Entities',\
								pickle_attrs='DataSet/semEval/semevalLaptop-Attributes', labels=labels, scheme=[True, True, False], one_hot=True)

sentiment_labels, sentiment_labeling = preprocess.encode_labels(pickle_entities='DataSet/semEval/semevalLaptop-Entities',\
								pickle_attrs='DataSet/semEval/semevalLaptop-Attributes', labels=labels, scheme=[False, False, True], one_hot=True)

print(aspect_labeling)
dataset = {}
dataset['name'] = 'semeval_laptops'
dataset['features'] = data
dataset['type'] = tf.string
dataset['tasks'] = [{'name' : 'aspects', 'features' : aspect_labels, 'type': tf.float32 }]#,\
										#{'name': 'sentiment', 'features': sentiment_labels, 'type':tf.float32 }]

datasets.append(dataset)


iterator, handles, handle, sess = build_tf_dataset(datasets)

handles = sess.run(handles)
print(handles)
for i in range(10):
	line = sess.run(iterator.get_next(), feed_dict={handle: handles[0]})
	print(line)
	print('\n-----------------')
	line = sess.run(iterator.get_next(), feed_dict={handle: handles[1]})
	print(line)
	print('\n-----------eof-----------------')

# test_data, test_labels = preprocess.flatten_multi_label('DataSet/semEval/semevalRestaurant_test-rawTextData',\
# 																												'DataSet/semEval/semevalRestaurant_test-rawTextLabels',)
#
# test_labels, test_labeling = preprocess.encode_labels(pickle_entities='DataSet/semEval/semevalRestaurant_test-Entities',\
# 								pickle_attrs='DataSet/semEval/semevalRestaurant_test-Attributes', labels=test_labels, scheme=[True, True, False], one_hot=True, labeling=labeling)

# test_vecs = openAI_transform(test_data)
# with open('FeatureExtrac/semevalRestaurants-openAI-data', 'rb') as fp:
# 	data_vecs = pickle.load(fp)
#
#


# for i in range(len(labels)):
# 	labels[i] = labeling[labels[i]]

# wdata = data_wrap(data_vecs)
# auto = autoencoder(wdata, num_step=12000)
#
# auto.train()
#
# reduced_data = auto.encode(data_vecs)
# reduced_data = np.squeeze(reduced_data)
# print(np.asarray(reduced_data).shape)
# print(labels)
# print('starting learning')
# supervised(data, data_vecs, labels, test_vecs, test_labels)

# print('\n-----------------------------------')
# reduced_data = np.transpose(reduced_data)
# supervised(data, reduced_data, labels)
