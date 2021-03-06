from Learning.multipath_tf_dataset import *
from utils import  *
import numpy as np
from DataSet.dataset_pipelines import *
from Learning.config_generator import *
from FeatureExtrac.openAI_transfrom import *
import pickle


#
rest_dict = SemEvalPipe(single_label=False, data='Restaurant', subtask=[True, False, False])
rest_dict2 = SemEvalPipe(single_label=False,data='Restaurant', subtask=[False, True, False])
rest_dict3 = SemEvalPipe(single_label=False,data='Restaurant', subtask=[True, True, False], entity_pred=True, attr_pred=True)
rest_dict4 = SemEvalPipe(data='Restaurant', subtask=[True, True, False])

# lap_dict = SemEvalPipe(single_label=False, data='Laptop', subtask=[True, False, False], noisy=True)
# lap_dict2 = SemEvalPipe(single_label=False,data='Laptop', subtask=[False, True, False], noisy=True)
# lap_dict3 = SemEvalPipe(single_label=False,data='Laptop', subtask=[True, True, False], attr_pred=True, entity_pred=True, noisy=True)
# lap_dict4 = SemEvalPipe(data='Laptop', subtask=[True, True, False])

# supervised(lap_dict3['train_data'], lap_dict3['train_vecs'], lap_dict3['encoded_train_labels'],
# 					 lap_dict3['test_vecs'], lap_dict3['encoded_test_labels'] )



datasets=[]
dataset={}
dataset['name'] = 'semeval_rest'
dataset['holdout'] = 100
dataset['batch_size'] = 300
dataset['features'] = rest_dict3['train_vecs']
dataset['type'] = tf.float32
dataset['tasks'] = [{'name' : 'rest_aspects', 'features' : rest_dict3['encoded_train_labels'], 'type': tf.float32}]
datasets.append(dataset)

# dataset={}
# dataset['name'] = 'semeval_lap'
# dataset['holdout'] = 100
# dataset['batch_size'] = 600
# dataset['features'] = lap_dict3['train_vecs']
# dataset['type'] = tf.float32
# dataset['tasks'] = [{'name' : 'lap_entities', 'features' : lap_dict3['encoded_train_labels'], 'type': tf.float32}]
# datasets.append(dataset)


paths = config_single_graph()
params={}
params['train_iter'] = 1500
M = TfMultiPathClassifier(datasets, paths, params)

M.save()
M.train()



x = M.get_prediciton('rest_aspects', rest_dict3['test_vecs'])
y = M.get_prediciton('rest_aspects', rest_dict3['train_vecs'])

multi_label_metrics(y, rest_dict3['train_labels'], rest_dict3['encoded_train_labels'], rest_dict3['labeling'], rest_dict3['train_data'])
multi_label_metrics(x, rest_dict3['test_labels'], rest_dict3['encoded_test_labels'], rest_dict3['labeling'], rest_dict3['test_data'], mute=False)


# x = M.get_raw_prediciton('rest_aspects', rest_dict2['test_vecs'])
# y = M.get_raw_prediciton('rest_aspects', rest_dict2['train_vecs'])
# 
# with open('rest_attrs_train_predictions_77', 'wb') as fp:
# 	pickle.dump(y,fp)
# with open('rest_attrs_test_predictions_77', 'wb') as fp:
# 	pickle.dump(x,fp)


