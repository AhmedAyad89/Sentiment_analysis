from Learning.multipath_tf_dataset import *
from utils import  *
from sklearn_pipeline import *
import numpy as np
from DataSet.dataset_pipelines import *
from Learning.config_generator import *




rest_dict3 = SemEvalPipe(single_label=False,data='Restaurant', subtask=[True, True, False])
lap_dict3 = SemEvalPipe(single_label=False,data='Laptop', subtask=[True, True, False])

org_dict = Hannah_pipe(single=False, scheme=[True, True], rel_filter=True)


datasets=[]
dataset={}
dataset['name'] = 'semeval_rest'
dataset['holdout'] = 200
dataset['batch_size'] = 200
dataset['features'] = rest_dict3['train_vecs']
dataset['type'] = tf.float32
dataset['tasks'] = [{'name' : 'rest_aspects', 'features' : rest_dict3['encoded_train_labels'], 'type': tf.float32}]
datasets.append(dataset)

dataset={}
dataset['name'] = 'semeval_lap'
dataset['holdout'] = 200
dataset['batch_size'] = 200
dataset['features'] = lap_dict3['train_vecs']
dataset['type'] = tf.float32
dataset['tasks'] = [{'name' : 'lap_aspects', 'features' : lap_dict3['encoded_train_labels'], 'type': tf.float32}]
datasets.append(dataset)

dataset={}
dataset['name'] = 'hannah_organic'
dataset['holdout'] = 20
dataset['batch_size'] = 200
dataset['features'] = org_dict['train_vecs']
dataset['type'] = tf.float32
dataset['tasks'] = [{'name' : 'org_aspects', 'features' : org_dict['encoded_train_labels'], 'type': tf.float32 }]
datasets.append(dataset)

paths = config_joint_graph()
params={}
params['train_iter'] = 3000
M = TfMultiPathClassifier(datasets, paths, params)

M.save()
M.train()

x = M.get_prediciton('org_aspects', org_dict['test_vecs'])
y = M.get_prediciton('org_aspects', org_dict['train_vecs'])

print(len(org_dict['test_labels']), len(org_dict['encoded_test_labels']))
multi_label_metrics(y, org_dict['train_labels'], org_dict['encoded_train_labels'], org_dict['labeling'], org_dict['train_data'])
multi_label_metrics(x, org_dict['test_labels'], org_dict['encoded_test_labels'], org_dict['labeling'], org_dict['test_data'], mute=False)
#
# 
# x = M.get_prediciton('rest_aspects', rest_dict3['test_vecs'])
# y = M.get_prediciton('rest_aspects', rest_dict3['train_vecs'])
# 
# multi_label_metrics(y, rest_dict3['train_labels'], rest_dict3['encoded_train_labels'], rest_dict3['labeling'], rest_dict3['train_data'])
# multi_label_metrics(x, rest_dict3['test_labels'], rest_dict3['encoded_test_labels'], rest_dict3['labeling'], rest_dict3['test_data'])
# 
# 
# x = M.get_prediciton('lap_aspects', lap_dict3['test_vecs'])
# y = M.get_prediciton('lap_aspects', lap_dict3['train_vecs'])
# 
# multi_label_metrics(y, lap_dict3['train_labels'], lap_dict3['encoded_train_labels'], lap_dict3['labeling'], lap_dict3['train_data'])
# multi_label_metrics(x, lap_dict3['test_labels'], lap_dict3['encoded_test_labels'], lap_dict3['labeling'], lap_dict3['test_data'])
