from Learning.multipath_tf_dataset import *
from visual import *
from utils import  *
from sklearn_pipeline import *
import tensorflow as tf
import os
import numpy as np
from DataSet.dataset_pipelines import *
from sklearn import metrics, preprocessing
from Learning.config_generator import *


ds_dict = SemEvalPipe(data='Restaurant', subtask=[True, False, False])
ds_dict2 = SemEvalPipe(single_label=True,data='Restaurant', subtask=[False, True, False])
datasets=[]


print(len(ds_dict['encoded_train_labels'][0]))
print(len(ds_dict2['encoded_train_labels'][0]))

dataset={}
dataset['name'] = 'semeval_laptops'
# dataset['holdout'] = 200
dataset['features'] = ds_dict['train_vecs']
dataset['type'] = tf.float32
dataset['tasks'] = [{'name' : 'entities', 'features' : ds_dict['encoded_train_labels'], 'type': tf.float32 }
										,{'name' : 'attributes', 'features' : ds_dict2['encoded_train_labels'], 'type': tf.float32 } ]

# dataset={}
# dataset['name'] = 'hannah_organic'
# dataset['features'] = data_vecs
# dataset['type'] = tf.float32
# dataset['tasks'] = [{'name' : 'aspects', 'features' : aspect_labels, 'type': tf.float32 }]
#
datasets.append(dataset)

test_labels = ds_dict['test_labels']
test_data = ds_dict['test_data']
entity_labeling = ds_dict['labeling']
attribute_labeling = ds_dict2['labeling']
paths = config_graph()
params={}
params['train_iter'] = 800
M = TfMultiPathClassifier(datasets, paths, params)

M.save()
M.train()

x = M.get_prediciton('entities', ds_dict['test_vecs'])
i = M.get_prediciton('attributes', ds_dict['test_vecs'])
pred_labels = np.stack((x,i),1)

x = ds_dict['label_encoder'].transform(x)
i = ds_dict2['label_encoder'].transform(i)

pred = np.concatenate((x,i),1)
l = np.concatenate((ds_dict['encoded_test_labels'], ds_dict2['encoded_test_labels']), 1)

try:
	print('micro', metrics.f1_score(l, pred, average='micro'))
	print('macro', metrics.f1_score(l, pred, average='macro'))
	print('weight', metrics.f1_score(l, pred, average='weighted'))
except:
	print('x')
print('samples', metrics.f1_score(l, pred, average='samples'))


counter = 0
for i in range(len(test_labels)):
	# if x[i] == 0 and test_labels[i] == 0:
	# 	continue
	print(i, '- ', test_data[i])#, test_aspect_labels[i], '\n',x[i])
	print(entity_labeling[pred_labels[i][0]], attribute_labeling[pred_labels[i][1]] )
	print(ds_dict['test_labels'][i], ds_dict2['test_labels'][i])
	#print(y[i][test_aspect_labels[i]])
	counter += 1
	print('\n', '\n--------------------------\n')


print(counter)