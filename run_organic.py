from Learning.multipath_tf_dataset import *
from utils import  *
from sklearn_pipeline import *
import tensorflow as tf
import numpy as np
from DataSet.dataset_pipelines import *
from Learning.config_generator import *

entity_dict = Hannah_pipe(single=False, rel_filter=True, scheme=[True, False], bow_clusters=True, merged_attr=True, merged_ent=False, entity_pred=True, train_cutoff=700)
attr_dict = Hannah_pipe(single=False, rel_filter=True, scheme=[True, True], bow_clusters=True, merged_attr=True,  train_cutoff=700, entity_pred=True, attr_pred=True)


datasets=[]

print(len(entity_dict['encoded_train_labels'][0]))



dataset={}
dataset['name'] = 'hannah_organic'
dataset['holdout'] = 50
dataset['batch_size'] = 450
dataset['features'] = attr_dict['train_vecs']
dataset['type'] = tf.float32
dataset['tasks'] = [{'name' : 'entities', 'features' : attr_dict['encoded_train_labels'], 'type': tf.float32 }]
								
datasets.append(dataset)


paths = config_organic_graph()
params={}
params['train_iter'] = 1500
M = TfMultiPathClassifier(datasets,  paths, params)

M.save()
M.train()

x = M.get_prediciton('entities', attr_dict['test_vecs'])
y = M.get_prediciton('entities', attr_dict['train_vecs'])
multi_label_metrics(y, attr_dict['train_labels'], attr_dict['encoded_train_labels'], attr_dict['labeling'], attr_dict['train_data'])
multi_label_metrics(x, attr_dict['test_labels'], attr_dict['encoded_test_labels'], attr_dict['labeling'], attr_dict['test_data'], mute=False)

#
# x = M.get_raw_prediciton('entities', attr_dict['test_vecs'])
# y = M.get_raw_prediciton('entities', attr_dict['train_vecs'])
#
# with open('org_attrs_train_predictions_5', 'wb') as fp:
# 	pickle.dump(y,fp)
# with open('org_attrs_test_predictions_5', 'wb') as fp:
# 	pickle.dump(x,fp)

