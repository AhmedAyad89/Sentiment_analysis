from Learning.multipath_tf_dataset import *
from visual import *
from utils import *
from sklearn_pipeline import *
import tensorflow as tf
import os
import numpy as np
from DataSet.dataset_pipelines import *
from sklearn import metrics, preprocessing
from Learning.config_generator import *
from cove_transform import *
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Conv2D, Flatten, GaussianDropout, Dropout, concatenate
import keras
from Learning.conv_net import *
from keras.utils import plot_model
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from sklearn import metrics





lap_dict3 = SemEvalPipe(single_label=False, data='Laptop', subtask=[True, True, False])

with open('semEvalLaptop-glove-data', 'rb') as fp:
	glove_data = pickle.load(fp)
with open('semEvalLaptop-cove-data', 'rb') as fp:
	cove_data = pickle.load(fp)
with open('semEvalLaptop_test-cove-data', 'rb') as fp:
	test_cove_data = pickle.load(fp)
with open('semEvalLaptop_test-glove-data', 'rb') as fp:
	test_glove_data = pickle.load(fp)
print(glove_data.shape)

cut=35
glove_data = glove_data[:, :cut, :]
cove_data = cove_data[:, :cut, :]
test_cove_data = test_cove_data[:, :cut, :]
test_glove_data = test_glove_data[:, :cut, :]
print(glove_data.shape)

try:
	with open('semEvalLaptop-coveVocab', 'rb') as fp:
		vocab = pickle.load(fp)
	with open('semEvalLaptop-coveVocabIndex', 'rb') as fp:
		vocab_index = pickle.load(fp)
except:
	print('could not load vocab')
	vocab, vocab_index = build_embedding_matrices(lap_dict3['train_data']+lap_dict3['test_data'])

train_sentences = convert_to_indices(lap_dict3['train_data'], max_len=35, vocab_index=vocab_index )
test_sentences = convert_to_indices(lap_dict3['test_data'], max_len=35, vocab_index=vocab_index )



model = keras_conv_net(vocab_size= len(vocab), embd_matrix=vocab)
model.fit([train_sentences, cove_data, lap_dict3['train_vecs']], lap_dict3['encoded_train_labels'], epochs=2, batch_size=50)

x = model.predict([test_sentences, test_cove_data, lap_dict3['test_vecs']], batch_size=50)
y = np.round(x)
multi_label_metrics(y, lap_dict3['test_labels'], lap_dict3['encoded_test_labels'], lap_dict3['labeling'],
										lap_dict3['test_data'], mute=False)

x = model.predict([train_sentences, cove_data, lap_dict3['train_vecs']])
y = np.round(x)
multi_label_metrics(y, lap_dict3['train_labels'], lap_dict3['encoded_train_labels'], lap_dict3['labeling'],
										lap_dict3['train_data'], mute=True)



#####################################################################################################################
# test_glove_sentences, test_cove_sentences = cove_pipeline\
# 	(lap_dict3['test_data'], glove_dmp= 'semEvalLaptop_test-glove-data', cove_dmp='semEvalLaptop_test-cove-data')


# combined = np.concatenate((glove_data, cove_data), axis=2)
# combined_test = np.concatenate((test_glove_data, test_cove_data), axis=2)
#
# model = Sequential()
# model.add(Conv1D(filters=256, kernel_size=5, activation='relu',kernel_regularizer=keras.regularizers.l1(0.0000005), input_shape=(cut, 900)))
# print(model.layers[-1].output_shape)
# model.add(MaxPooling1D(pool_size=2))
# print(model.layers[-1].output_shape)
# model.add(Flatten())
# model.add(Dense(units=256, activation='relu'))
# model.add(Dense(units=88, activation='sigmoid'))
# model.compile(optimizer='rmsprop',
# 							loss='binary_crossentropy')
#
#
# model.fit(combined , lap_dict3['encoded_train_labels'], epochs=40, batch_size=50)

