import pickle
from itertools import compress
import numpy as np
from sklearn.preprocessing import LabelEncoder,OneHotEncoder, normalize,MultiLabelBinarizer, LabelBinarizer
from FeatureExtrac.openAI_transfrom import *
#takes data as a list of lists and returns linear list
#takes multi-labels as a list of lists of lists and returns a linear single-label list
#takes only the first label of each sentence
def flatten_single_label(pickle_data=None, pickle_labels =None, data=None, labels=None):
	if(pickle_data is not None):
		with open(pickle_data, 'rb') as fp:
			data = pickle.load(fp)
	if (pickle_labels is not None):
		with open(pickle_labels, 'rb') as lb:
			labels = pickle.load(lb)

	if (data is None) or (labels is None):
		print('Not enough arguments passed')
		return 0

	new_data = []
	new_labels=[]
	for i,review in enumerate(data):
		for j, sentence in enumerate(review):
			new_data.append(sentence)
			new_labels.append(labels[i][j][0])
			#print(sentence, labels[i][j][0], '\n----------\n')

	# for i in range(len(new_data)):
	# 	print(new_data[i], new_labels[i])
	# 	print('\n-----------------\n',i)
	print(len(new_data), len(new_labels))
	return new_data, new_labels

# this one returns labels as a list of lists
def flatten_multi_label(pickle_data=None, pickle_labels =None, data=None, labels=None):
	if(pickle_data is not None):
		with open(pickle_data, 'rb') as fp:
			data = pickle.load(fp)
	if (pickle_labels is not None):
		with open(pickle_labels, 'rb') as lb:
			labels = pickle.load(lb)

	if (data is None) or (labels is None):
		print('Not enough arguments passed')
		return 0

	new_data = []
	new_labels=[]
	review_lengths =[]
	for i, label in enumerate(labels):
		for j, e in enumerate(label):
			if e == [['NA', 'NA', 'NA']]:
				labels[i][j] = []
	for i,review in enumerate(data):
		l = 0
		for j, sentence in enumerate(review):
			l+=1
			new_data.append(sentence)
			new_labels.append(labels[i][j])
		review_lengths.append(l)
	return new_data, new_labels

#returns labeling for inversion of the encoding
#scheme defines which parts(entity, attribute, polarity) will be encoded
#if modular set to false, each unique combination is given a label, otherwise each part gets its own labeling. not implemented yet
def encode_labels(pickle_labels= None, pickle_entities=None, pickle_attrs=None, labels=None, label_set=None,  \
									 one_hot= False, modular=True,scheme=[True,True,True], labeling=None, encoder=None):
	if (pickle_labels is not None):
		with open(pickle_labels, 'rb') as lb:
			labels = pickle.load(lb)
	if (pickle_entities is not None):
		with open(pickle_entities, 'rb') as lb:
			entities = pickle.load(lb)
	if (pickle_attrs is not None):
		with open(pickle_attrs, 'rb') as lb:
			attrs = pickle.load(lb)

	polarity = ['negative', 'neutral', 'positive','NA']
	cat=[]
	pol=[]
	if (labeling is None):
		labeling=[]
	L=[]
	single = True
	for i in labels:
		if any(isinstance(a, list) for a in i) :
			single =False
			break

	if single:	#single label
		for e in labels:
			#print(e)
			#e[0] = entities.index(e[0])
			#e[1] = attrs.index(e[1])
			#e[2] = polarity.index(e[2]) - 1
			filtered = list(compress(e,scheme))
			#print(filtered)
			if (filtered not in labeling):
				labeling.append(filtered)
			L.append(labeling.index(filtered))
			# print('-----------single--------------\n')
		if (one_hot):
			if encoder is None:
				encoder = LabelBinarizer()
				L = encoder.fit_transform(np.asarray(L).reshape(len(L), 1))
			else:
				L = encoder.transform(np.asarray(L).reshape(len(L), 1))
		return L, encoder, labeling
	else:																		#multi-label
		for s in labels:
			S=[]
			for e in s:
				#e[0]= entities.index(e[0])
				#e[1] = attrs.index(e[1])
				#e[2]=polarity.index(e[2])-1
				filtered = list(compress(e, scheme))
				#print(filtered)
				if (filtered not in labeling):
					labeling.append(filtered)
				S.append(labeling.index(filtered))
			L.append(S)
			# print('-------------------------\n')
		if (one_hot):
			# for lab in L:
			# 	temp = np.zeros(len(labeling))
			# 	for i in lab:
			# 		temp[i] = 1
			# 	oh.append( temp )
			sum=0
			for entry in L:
				sum += len(entry)
			print('avg length', sum/ len(L))
			if encoder is None:
				encoder = MultiLabelBinarizer()
				L = encoder.fit_transform(L)
			else:
				L = encoder.transform(L)
			return L, encoder, labeling

	return L,0, labeling

def semEval_review_level_text(pickle_data=None, pickle_dmp=None):
	if(pickle_data is not None):
		with open(pickle_data, 'rb') as fp:
			data = pickle.load(fp)
		review_lengths=[]
		joined_reviews =[]
		for review in data:
			review_lengths.append(len(review))
			review = ' '.join(review)
			joined_reviews.append(review)

		vecs = openAI_transform(joined_reviews)
		context_vecs = []
		m = np.max(review_lengths)
		for c, i in enumerate(review_lengths):
			for j in range(i):
				loc = np.zeros(m)
				loc[j] = 1
				v = np.concatenate((vecs[c], loc), axis=0)
				context_vecs.append(v)

		with open(pickle_dmp, 'wb') as fp:
			pickle.dump(context_vecs, fp)


if __name__ == '__main__':
	print('x')
	# data, labels = flatten_multi_label('semEval\\semevalLaptop-rawTextData','semEval\\semevalLaptop-rawTextLabels')
	# encode_labels(pickle_entities='semEval\\semevalLaptop-Entities',\
	# 							pickle_attrs='semEval\\semevalLaptop-Attributes', labels=labels, scheme=[True, True, False], one_hot=True)
	semEval_review_level_text('semEval\\semevalLaptop-rawTextData', 'semevalLaptop-openAI-data-reviewLevel')