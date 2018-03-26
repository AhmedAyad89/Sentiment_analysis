from Learning.multipath_tf_dataset import *
from utils import *
from sklearn_pipeline import *
import tensorflow as tf
import numpy as np
from DataSet.dataset_pipelines import *
from Learning.config_generator import *

sent_dict = Hannah_pipe(single=True, rel_filter=True, scheme=[False, False, True], train_cutoff=700)

for i in range(300):
	print(sent_dict['train_data'][i])
	print(sent_dict['train_labels'][i])
	print(sent_dict['encoded_train_labels'][i])

datasets = []

print(len(sent_dict['test_vecs']))
print(len(sent_dict['encoded_test_labels']))
dataset = {}
dataset['name'] = 'hannah_organic'
dataset['holdout'] = 50
dataset['batch_size'] = 100
dataset['features'] = sent_dict['train_vecs']
dataset['type'] = tf.float32
dataset['tasks'] = [{'name': 'sent', 'features': sent_dict['encoded_train_labels'], 'type': tf.float32}]

datasets.append(dataset)

paths = config_sentiment_graph()
params = {}
params['train_iter'] = 1500
M = TfMultiPathClassifier(datasets, paths, params)

M.save()
M.train()

x = M.get_prediciton('sent', sent_dict['test_vecs'])
y = M.get_prediciton('sent', sent_dict['train_vecs'])
multi_label_metrics(y, sent_dict['train_labels'], sent_dict['encoded_train_labels'], sent_dict['labeling'],
										sent_dict['train_data'])
multi_label_metrics(x, sent_dict['test_labels'], sent_dict['encoded_test_labels'], sent_dict['labeling'],
										sent_dict['test_data'], mute=False)

#
# x = M.get_raw_prediciton('entities', attr_dict['test_vecs'])
# y = M.get_raw_prediciton('entities', attr_dict['train_vecs'])
#
# with open('org_attrs_train_predictions_5', 'wb') as fp:
# 	pickle.dump(y,fp)
# with open('org_attrs_test_predictions_5', 'wb') as fp:
# 	pickle.dump(x,fp)

