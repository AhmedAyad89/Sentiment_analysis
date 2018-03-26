from Learning.multipath_tf_dataset import *
from utils import  *
import numpy as np
from DataSet.dataset_pipelines import *
from Learning.config_generator import *
import pickle


rest_dict = SemEvalPipe_sentiment(data='Laptop')


# lap_dict = SemEvalPipe(single_label=False, data='Laptop', subtask=[True, False, False], noisy=True)
# lap_dict2 = SemEvalPipe(single_label=False,data='Laptop', subtask=[False, True, False], noisy=True)
# lap_dict3 = SemEvalPipe(single_label=False,data='Laptop', subtask=[True, True, False], attr_pred=True, entity_pred=True, noisy=True)
# lap_dict4 = SemEvalPipe(data='Laptop', subtask=[True, True, False])

# for i in range(500):
# 	print(rest_dict['train_data'][i], rest_dict['train_labels'][i], rest_dict['encoded_train_labels'][i])
# 	print(rest_dict['test_data'][i], rest_dict['test_labels'][i], rest_dict['encoded_test_labels'][i])

datasets=[]
dataset={}
dataset['name'] = 'semeval_rest'
# dataset['holdout'] = 100
dataset['batch_size'] = 50
dataset['features'] = rest_dict['train_vecs']
dataset['type'] = tf.float32
dataset['tasks'] = [{'name' : 'sent', 'features' : rest_dict['encoded_train_labels'], 'type': tf.float32}]
datasets.append(dataset)

# dataset={}
# dataset['name'] = 'semeval_lap'
# dataset['holdout'] = 100
# dataset['batch_size'] = 600
# dataset['features'] = lap_dict3['train_vecs']
# dataset['type'] = tf.float32
# dataset['tasks'] = [{'name' : 'lap_entities', 'features' : lap_dict3['encoded_train_labels'], 'type': tf.float32}]
# datasets.append(dataset)


paths = config_sentiment_graph()
params={}
params['train_iter'] = 4500
M = TfMultiPathClassifier(datasets, paths, params)

M.save()
M.train()



x = M.get_prediciton('sent', rest_dict['test_vecs'])
y = M.get_prediciton('sent', rest_dict['train_vecs'])

multi_label_metrics(y, rest_dict['train_labels'], rest_dict['encoded_train_labels'], rest_dict['labeling'], rest_dict['train_data'])
multi_label_metrics(x, rest_dict['test_labels'], rest_dict['encoded_test_labels'], rest_dict['labeling'], rest_dict['test_data'], mute=False)


# x = M.get_raw_prediciton('rest_aspects', rest_dict2['test_vecs'])
# y = M.get_raw_prediciton('rest_aspects', rest_dict2['train_vecs'])
#
# with open('rest_attrs_train_predictions_77', 'wb') as fp:
# 	pickle.dump(y,fp)
# with open('rest_attrs_test_predictions_77', 'wb') as fp:
# 	pickle.dump(x,fp)


