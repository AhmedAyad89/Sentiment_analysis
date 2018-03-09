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



org_dict = Hannah_pipe(scheme=[False, True])
datasets=[]




print(len(org_dict['encoded_train_labels'][0]))


# supervised(org_dict['train_data'], org_dict['train_vecs'], org_dict['encoded_train_labels'],
# 					 org_dict['test_vecs'], org_dict['encoded_test_labels'] )

# single_train = []
# single_test = []
# for l in ds_dict3['encoded_train_labels']:
# 	if np.linalg.norm(l, ord=0) == 0:
# 		n = np.concatenate([l, [1]])
# 		single_train.append(n)
# 	else:
# 		n =  l / np.linalg.norm(l, ord=0)
# 		n = np.concatenate([n, [0]])
# 		single_train.append(n)
# #
# for i in range(10):
# 	print(single_train)

# for l in ds_dict3['encoded_test_labels']:
# 	single_test.append( l / np.linalg.norm(l, ord=0) )
#



x =input()
# pred = [y[:-1] for y in x]
# pred = np.asarray(pred, dtype=np.int32)
pred = x
l = org_dict['encoded_test_labels']
# print(pred[0], l[0])
# print(len(pred[0]))
try:
	print('micro', metrics.f1_score(l, pred, average='micro'))
	# print('macro', metrics.f1_score(l, pred, average='macro'))
	# print('weight', metrics.f1_score(l, pred, average='weighted'))
except Exception as e:
	print(e)
print('f1 samples', metrics.f1_score(l, pred, average='samples'))
print('precision samples', metrics.precision_score(l, pred, average='samples'))
print('recall samples', metrics.recall_score(l, pred, average='samples'))

xp = np.where(x == 1)
counter = 0
for i in range(len(test_labels)):
	# if x[i] == 0 and test_labels[i] == 0:
	# 	continue
	print(i, '- ', test_data[i])#, test_aspect_labels[i], '\n',x[i])
	try:
		if(xp[i] == 88):
			print('NA')
		else:
			print([aspect_labeling[entry] for entry in xp[i]])
	except:
		print((aspect_labeling[xp[i]]))
	print(org_dict['test_labels'][i])
	#print(y[i][test_aspect_labels[i]])
	counter += 1
	print('\n', '\n--------------------------\n')


print(counter)