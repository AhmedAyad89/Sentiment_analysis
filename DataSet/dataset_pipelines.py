from DataSet import preprocess
import pickle
from FeatureExtrac.openAI_transfrom import openAI_transform
from FeatureExtrac.bag_of_words import *
from utils import  *
from itertools import compress
from FeatureExtrac.bag_of_words import *
from copy import deepcopy
import regex as re

#subtask  = [aspect, sentiment, both]
def SemEvalPipe(data='Laptop', subtask=[True, True, True], test=True, single_label = True, one_hot= True, review_context=False, noisy=False, attr_pred=False, entity_pred=False  ):
	datasets=[]


	if single_label:
		train_data, train_labels = preprocess.flatten_single_label('DataSet/semEval/semeval'+data+'-rawTextData',
																									'DataSet/semEval/semeval'+data+'-rawTextLabels',)
		test_data, test_labels = preprocess.flatten_single_label('DataSet/semEval/semeval'+data+'_test-rawTextData',
																									'DataSet/semEval/semeval'+data+'_test-rawTextLabels',)
	else:
		train_data, train_labels= preprocess.flatten_multi_label('DataSet/semEval/semeval'+data+'-rawTextData',
																									'DataSet/semEval/semeval'+data+'-rawTextLabels',)

		test_data, test_labels = preprocess.flatten_multi_label('DataSet/semEval/semeval'+data+'_test-rawTextData',
																									'DataSet/semEval/semeval'+data+'_test-rawTextLabels',)

	print(len(train_labels + test_labels))
	encoded_labels, encoder, labeling = preprocess.encode_labels\
		(labels=train_labels + test_labels, scheme=subtask, one_hot=one_hot)

	if noisy:
		noisy_labels = np.zeros(shape=encoded_labels.shape)
		for i in range(len(encoded_labels)):
			for j in range(len(encoded_labels[i])):
				if encoded_labels[i][j] == 1:
					noisy_labels[i][j] = 0.95
				else:
					noisy_labels[i][j] = 0.01
		noisy_train_labels = noisy_labels[:len(train_labels)]

	encoded_train_labels = encoded_labels[:len(train_labels)]
	encoded_test_labels = encoded_labels[len(train_labels):]

	if (one_hot):
		inverted_labels = encoder.inverse_transform(encoded_labels)
	else:
		inverted_labels = encoded_labels
	reoriginal=[]
	#Sanity check
	if not single_label:
		for i in inverted_labels:
			tmp=[]
			for e in i:
				tmp.append(labeling[e])
			reoriginal.append(tmp)

		for i in range(len(train_labels)):
			# print(train_data[i], train_labels[i], reoriginal[i])
			s = set
			for j, e in enumerate(train_labels[i]):
				assert  list(compress(e, subtask)) in reoriginal[i]

		for i in range(len(train_labels), len(reoriginal)):
			j = i- len(train_labels)
			# print(test_data[j], test_labels[j], reoriginal[i])
			s = set
			for j, e in enumerate(test_labels[j]):
				assert  list(compress(e, subtask)) in reoriginal[i]

		print('assertion passed')
	else:
		for e in inverted_labels:
			reoriginal.append(labeling[e])

		for i in range(len(train_labels)):
			# print(train_data[i], train_labels[i], reoriginal[i])
			assert list(compress(train_labels[i], subtask)) == reoriginal[i]

		for i in range(len(train_labels), len(reoriginal)):
			j = i - len(train_labels)
			#print(test_data[j], test_labels[j], reoriginal[i])
			assert list(compress(test_labels[j], subtask)) == reoriginal[i]

		print('assertion passed')

	try:
		with open('FeatureExtrac//semeval' + data + '-openAI-data', 'rb') as fp:
			train_vecs = pickle.load(fp)
	except Exception as e:
		print(e)
		train_vecs = openAI_transform(train_data, dmp_name='FeatureExtrac//semeval' + data + '-openAI-data')

	try:
		with open('FeatureExtrac//semeval' + data + '_test-openAI-data', 'rb') as fp:
			test_vecs = pickle.load(fp)
	except:
		test_vecs = openAI_transform(test_data, dmp_name='FeatureExtrac//semeval' + data + '_test-openAI-data')

	if review_context:
		with open('FeatureExtrac/semeval'+data+'-openAI-data-reviewLevel', 'rb') as  fp:
			train_context_vecs = pickle.load(fp)
		with open('FeatureExtrac/semeval'+data+'_test-openAI-data-reviewLevel', 'rb') as  fp:
			test_context_vecs = pickle.load(fp)
		train_vecs = np.concatenate((train_vecs, train_context_vecs), axis=1)
		test_vecs =  np.concatenate((test_vecs, test_context_vecs), axis=1)

	if attr_pred:
		with open('rest_attrs_train_predictions_77', 'rb') as fp:
			attr_train_predictions = pickle.load(fp)
		with open('rest_attrs_test_predictions_77', 'rb') as fp:
			attr_test_predictions = pickle.load(fp)
		train_vecs = np.concatenate((train_vecs, attr_train_predictions), axis=1)
		test_vecs =  np.concatenate((test_vecs, attr_test_predictions), axis=1)

	if entity_pred:
		with open('rest_entities_train_predictions_77', 'rb') as fp:
			entities_train_predictions = pickle.load(fp)
		with open('rest_entities_test_predictions_77', 'rb') as fp:
			entities_test_predictions = pickle.load(fp)
		train_vecs = np.concatenate((train_vecs, entities_train_predictions), axis=1)
		test_vecs = np.concatenate((test_vecs, entities_test_predictions), axis=1)

	if single_label:
		train_labels = [list(compress(x, subtask)) for x in train_labels]
		test_labels =  [list(compress(i, subtask)) for i in test_labels]
	else:
		new_test=[]
		for i, entry in enumerate(test_labels):
			tmp=[]
			for a in entry:
				tmp.append(list(compress(a, subtask)))
			new_test.append(tmp)
		new_train=[]
		for i, entry in enumerate(train_labels):
			tmp=[]
			for a in entry:
				tmp.append(list(compress(a, subtask)))
			new_train.append(tmp)
		train_labels =new_train
		test_labels =new_test
	dict = {}
	if noisy:
		dict['noisy_labels'] = noisy_train_labels
	dict['train_data'] = train_data
	dict['train_labels'] = train_labels
	dict['train_vecs'] = train_vecs
	dict['encoded_train_labels'] = encoded_train_labels
	dict['test_data'] = test_data
	dict['test_labels'] = test_labels
	dict['test_vecs'] = test_vecs
	dict['encoded_test_labels'] = encoded_test_labels
	dict['labeling'] = labeling
	dict['label_encoder'] = encoder
	return dict

def SemEvalPipe_sentiment(data='Laptop', subtask=[False, False, True], test=True, single_label = True, one_hot= True, review_context=False, noisy=False, attr_pred=False, entity_pred=False  ):
	datasets=[]

	train_data, train_labels= preprocess.flatten_multi_label('DataSet/semEval/semeval'+data+'-rawTextData',
																								'DataSet/semEval/semeval'+data+'-rawTextLabels',)

	test_data, test_labels = preprocess.flatten_multi_label('DataSet/semEval/semeval'+data+'_test-rawTextData',
																								'DataSet/semEval/semeval'+data+'_test-rawTextLabels',)
	new_data = []
	new_labels = []
	for i, sentence in enumerate(train_data):
		for label in train_labels[i]:
			l =  ' '.join(label[:2])
			l = l.lower()
			l = l.replace('_', ' ')
			new_data.append(sentence +' '+ l)
			new_labels.append(label[2:])
	new_test_data = []
	new_test_labels = []
	for i, sentence in enumerate(test_data):
		for label in test_labels[i]:
			l =  ' '.join(label[:2])
			l = l.lower()
			l = l.replace('_', ' ')
			new_test_data.append(sentence +' '+ l)
			new_test_labels.append(label[2:])
	train_data = new_data
	train_labels = new_labels
	test_data = new_test_data
	test_labels = new_test_labels


	print(len(train_labels + test_labels))
	encoded_labels, encoder, labeling = preprocess.encode_labels\
		(labels=train_labels + test_labels, scheme=[True], one_hot=one_hot)


	encoded_train_labels = encoded_labels[:len(train_labels)]
	encoded_test_labels = encoded_labels[len(train_labels):]

	# if (one_hot):
	# 	inverted_labels = encoder.inverse_transform(encoded_labels)
	# else:
	# 	inverted_labels = encoded_labels
	# reoriginal=[]
	# #Sanity check
	# if not single_label:
	# 	for i in inverted_labels:
	# 		tmp=[]
	# 		for e in i:
	# 			tmp.append(labeling[e])
	# 		reoriginal.append(tmp)
	#
	# 	for i in range(len(train_labels)):
	# 		# print(train_data[i], train_labels[i], reoriginal[i])
	# 		s = set
	# 		for j, e in enumerate(train_labels[i]):
	# 			assert  list(compress(e, subtask)) in reoriginal[i]
	#
	# 	for i in range(len(train_labels), len(reoriginal)):
	# 		j = i- len(train_labels)
	# 		# print(test_data[j], test_labels[j], reoriginal[i])
	# 		s = set
	# 		for j, e in enumerate(test_labels[j]):
	# 			assert  list(compress(e, subtask)) in reoriginal[i]
	#
	# 	print('assertion passed')
	# else:
	# 	for e in inverted_labels:
	# 		reoriginal.append(labeling[e])
	#
	# 	for i in range(len(train_labels)):
	# 		# print(train_data[i], train_labels[i], reoriginal[i])
	# 		assert list(compress(train_labels[i], subtask)) == reoriginal[i]
	#
	# 	for i in range(len(train_labels), len(reoriginal)):
	# 		j = i - len(train_labels)
	# 		#print(test_data[j], test_labels[j], reoriginal[i])
	# 		assert list(compress(test_labels[j], subtask)) == reoriginal[i]
	#
	# 	print('assertion passed')

	try:
		with open('FeatureExtrac//semeval' + data + '-sent-openAI-data', 'rb') as fp:
			train_vecs = pickle.load(fp)
	except Exception as e:
		print(e)
		train_vecs = openAI_transform(train_data, dmp_name='FeatureExtrac//semeval' + data + '-sent-openAI-data')

	try:
		with open('FeatureExtrac//semeval' + data + '-sent_test-openAI-data', 'rb') as fp:
			test_vecs = pickle.load(fp)
	except:
		test_vecs = openAI_transform(test_data, dmp_name='FeatureExtrac//semeval' + data + '-sent_test-openAI-data')


	dict = {}
	dict['train_data'] = train_data
	dict['train_labels'] = train_labels
	dict['train_vecs'] = train_vecs
	dict['encoded_train_labels'] = encoded_train_labels
	dict['test_data'] = test_data
	dict['test_labels'] = test_labels
	dict['test_vecs'] = test_vecs
	dict['encoded_test_labels'] = encoded_test_labels
	dict['labeling'] = labeling
	dict['label_encoder'] = encoder
	return dict

		
def Hannah_pipe(train_cutoff = 700, single=True, rel_filter =True, truncated_labels=True, bow_clusters=False, scheme=[True, True], merged_attr=False, merged_ent=False, entity_pred=False, attr_pred=False ):
	
	entity_dict, attr_dict = parse_hannah_legend('DataSet//organic//legend.csv')
	data, entities, attr, sent = parse_hannah_csv('DataSet//organic//19-03-05_comments_HD.csv')

	sent_dict = { 'p': 'Positive', 'n': 'Negative', '0':'Neutral', '':'not relevant' }
	print(entity_dict)
	print(attr_dict)
	if merged_attr:
		attr_dict['ll'] = attr_dict['q']
		attr_dict['s'] = attr_dict['q']
		attr_dict['l'] = attr_dict['q']
		attr_dict['av'] = attr_dict['q']
		attr_dict['t'] = attr_dict['q']

		attr_dict['c'] = attr_dict['h']

		attr_dict['a'] = attr_dict['e']
		attr_dict['pp'] = attr_dict['e']

	if merged_ent:
		entity_dict['p'] = entity_dict['g']
		entity_dict['f'] = entity_dict['g']
		entity_dict['c'] = entity_dict['g']

		entity_dict['cp'] = entity_dict['cg']
		entity_dict['cf'] = entity_dict['cg']
		entity_dict['cc'] = entity_dict['cg']

	if single:
		for i,e in enumerate(entities):
			try:
				entities[i] = entity_dict[e[0]]
			except:
				print(i, e[0])
				entities[i] = entity_dict['rel']
		for i,e in enumerate(attr):
			try:
				attr[i] = attr_dict[e[0]]
			except:
				print(i, e[0])
				attr[i] = attr_dict['rel']
		for i,e in enumerate(sent):
			try:
				sent[i] = sent_dict[e[0]]
			except:
				print(i, e[0])
				sent[i] = sent_dict['0']
		labels = zip(entities, attr, sent)
		new=[]
		for i in labels:
			new.append(list(i))
		labels=new
	else:
		for i,l in enumerate(entities):
			for j, e in enumerate(l):
				try:
					entities[i][j] = entity_dict[e]
				except:
					print(i, e)
					entities[i][j] = entity_dict['rel']
		for i,l in enumerate(attr):
			for j, e in enumerate(l):
				try:
					attr[i][j] = attr_dict[e]
				except:
					print(i, e)
					attr[i][j] = attr_dict['rel']
		labels=[]
		new=[]
		for i, entry in enumerate(entities):
			labels.append(zip(entities[i], attr[i] ))
		for i in range(len(labels)):
			new.append([list(x) for x in labels[i]])
		labels=new


	new_data=[]
	new_labels=[]
	if single:
		none_label =[]
	else:
		none_label=[]
	if rel_filter:
		for i in range(len(data)):
			print(labels[i])
			if labels[i] == ['not relevant', 'not relevant','not relevant'] or ['not relevant', 'not relevant','not relevant'] in labels[i]:
				continue
			if labels[i] == ['relevant','relevant','relevant'] or labels[i] == ['relevant','relevant','not relevant'] or ['relevant','not relevant','not relevant'] in labels[i]:
				continue
			new_data.append(data[i])
			new_labels.append(labels[i])
		data = new_data
		labels = new_labels
		print('remaining', len(data), len(labels))
	elif truncated_labels:
		for i in range(len(data)):
			if labels[i] == ['not relevant', 'not relevant'] or ['not relevant', 'not relevant'] in labels[i]:
				labels[i] = none_label
			if labels[i] == ['relevant', 'relevant'] or ['relevant', 'relevant'] in labels[i]:
				labels[i] = none_label


	encoded_labels, encoder, labeling = preprocess.encode_labels(labels = labels, scheme=scheme, one_hot=True )
	inverted_labels= encoder.inverse_transform(encoded_labels)


	#sanity check
	# if single:
	# 	for i in range(len(data)):
	# 		# print(data[i], labels[i], labeling[inverted_labels[i]])
	# 		assert labeling[inverted_labels[i]] == labels[i]
	# else:
	# 	reoriginal=[]
	# 	for i in inverted_labels:
	# 		tmp=[]
	# 		for e in i:
	# 			tmp.append(labeling[e])
	# 		reoriginal.append(tmp)
	# 	for i in range(len(data)):
	# 		for j in range(len(labels[i])):
	# 			assert list(labels[i][j]) in reoriginal[i]

	if rel_filter:
		try:
			with open("FeatureExtrac//hannah-openAI-data-filtered", 'rb') as fp:
				data_vecs = pickle.load(fp)
		except:
			data_vecs = openAI_transform(data, dmp_name="FeatureExtrac//hannah-openAI-data-filtered")
	else:
		try:
			with open("FeatureExtrac//hannah-openAI-data", 'rb') as fp:
				data_vecs = pickle.load(fp)
		except:
			data_vecs = openAI_transform(data, dmp_name="FeatureExtrac//hannah-openAI-data")

	if bow_clusters:
		features = bow_clusters_features(data = data)
		data_vecs = np.concatenate((data_vecs, features), axis=1)

	print(len(data_vecs[0]))

	print(len(data_vecs))
	test_vecs = data_vecs[train_cutoff:]
	train_vecs = data_vecs[:train_cutoff]
	encoded_test_labels = encoded_labels[train_cutoff:]
	encoded_train_labels=encoded_labels[:train_cutoff]
	test_data = data[train_cutoff:]
	train_data = data[:train_cutoff]
	train_labels = labels[:train_cutoff]
	test_labels = labels[train_cutoff:]

	if attr_pred:
		with open('org_attrs_train_predictions_5', 'rb') as fp:
			attr_train_predictions = pickle.load(fp)
		with open('org_attrs_test_predictions_5', 'rb') as fp:
			attr_test_predictions = pickle.load(fp)
		train_vecs = np.concatenate((train_vecs, attr_train_predictions), axis=1)
		test_vecs =  np.concatenate((test_vecs, attr_test_predictions), axis=1)

	if entity_pred:
		with open('org_entities_train_predictions_9', 'rb') as fp:
			entities_train_predictions = pickle.load(fp)
		with open('org_entities_test_predictions_9', 'rb') as fp:
			entities_test_predictions = pickle.load(fp)
		train_vecs = np.concatenate((train_vecs, entities_train_predictions), axis=1)
		test_vecs = np.concatenate((test_vecs, entities_test_predictions), axis=1)

	dict = {}
	name= 'organic'
	dict['train_data'] = train_data
	dict['train_labels'] = train_labels
	dict['train_vecs'] = train_vecs
	dict['encoded_train_labels'] = encoded_train_labels
	dict['test_data'] = test_data
	dict['test_labels'] = test_labels
	dict['test_vecs'] = test_vecs
	dict['encoded_test_labels'] = encoded_test_labels
	dict['labeling'] = labeling
	dict['label_encoder'] = encoder
	return dict



if __name__ == '__main__':
	SemEvalPipe()