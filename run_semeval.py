from Learning.multipath_tf_dataset import *
from utils import  *
from sklearn_pipeline import *
import numpy as np
from DataSet.dataset_pipelines import *
from Learning.config_generator import *




ds_dict = SemEvalPipe(single_label=False, data='Laptop', subtask=[True, False, False])
ds_dict2 = SemEvalPipe(single_label=False,data='Laptop', subtask=[False, True, False])
ds_dict3 = SemEvalPipe(single_label=False,data='Laptop', subtask=[True, True, False])
ds_dict4 = SemEvalPipe(data='Laptop', subtask=[True, True, False])



datasets=[]
dataset={}
dataset['name'] = 'semeval_laptops'
dataset['holdout'] = 250
dataset['batch_size'] = 50
dataset['features'] = ds_dict['train_vecs']
dataset['type'] = tf.float32
dataset['tasks'] = [{'name' : 'entities', 'features' : ds_dict['encoded_train_labels'], 'type': tf.float32 }
										,{'name' : 'attributes', 'features' : ds_dict2['encoded_train_labels'], 'type': tf.float32 }
										,{'name' : 'aspects', 'features' : ds_dict3['encoded_train_labels'], 'type': tf.float32}]

datasets.append(dataset)


test_labels = ds_dict4['test_labels']
test_data = ds_dict4['test_data']

aspect_labeling = ds_dict4['labeling']

paths = config_single_graph()
params={}
params['train_iter'] = 20000
M = TfMultiPathClassifier(datasets, paths, params)

M.save()
M.train()

x = M.get_prediciton('aspects', ds_dict3['test_vecs'])
y = M.get_prediciton('aspects', ds_dict3['train_vecs'])



multi_label_metrics(y, ds_dict3['train_labels'], ds_dict3['encoded_train_labels'], ds_dict3['labeling'], ds_dict3['train_data'])
multi_label_metrics(x, ds_dict3['test_labels'], ds_dict3['encoded_test_labels'], ds_dict3['labeling'], ds_dict3['test_data'])
