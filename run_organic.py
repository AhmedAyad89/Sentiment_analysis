from Learning.multipath_tf_dataset import *
from utils import  *
from sklearn_pipeline import *
import tensorflow as tf
import numpy as np
from DataSet.dataset_pipelines import *
from Learning.config_generator import *

org_dict = Hannah_pipe(scheme=[False, True])
datasets=[]

print(len(org_dict['encoded_train_labels'][0]))

dataset={}
dataset['name'] = 'hannah_organic'
dataset['batch_size'] = 100
dataset['features'] = org_dict['train_vecs']
dataset['type'] = tf.float32
dataset['tasks'] = [{'name' : 'aspects', 'features' : org_dict['encoded_train_labels'], 'type': tf.float32 }]
datasets.append(dataset)


test_labels = org_dict['test_labels']
test_data = org_dict['test_data']
# entity_labeling = ds_dict['labeling']
# attribute_labeling = ds_dict2['labeling']
aspect_labeling = org_dict['labeling']

paths = config_single_graph()
params={}
params['train_iter'] = 2000
M = TfMultiPathClassifier(datasets, paths, params)

M.save()
M.train()

x = M.get_prediciton('aspects', org_dict['test_vecs'])
y = M.get_prediciton('aspects', org_dict['train_vecs'])
single_label_metrics(y, org_dict['train_labels'], org_dict['encoded_train_labels'], org_dict['labeling'], org_dict['train_data'])
single_label_metrics(x, org_dict['test_labels'], org_dict['encoded_test_labels'], org_dict['labeling'], org_dict['test_data'])
