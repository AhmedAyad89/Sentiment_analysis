from DataSet import preprocess
import pickle
from FeatureExtrac.openAI_transfrom import openAI_transform
from utils import  *
from itertools import compress


#subtask  = [aspect, sentiment, both]
def SemEvalPipe(data='Laptop', subtask = [True, True, True], test = True, single_label = True, one_hot= True  ):
	datasets=[]
	if single_label:
		train_data, train_labels = preprocess.flatten_single_label('DataSet/semEval/semeval'+data+'-rawTextData',
																									'DataSet/semEval/semeval'+data+'-rawTextLabels',)
		test_data, test_labels = preprocess.flatten_single_label('DataSet/semEval/semeval'+data+'_test-rawTextData',
																									'DataSet/semEval/semeval'+data+'_test-rawTextLabels',)
	else:
		train_data, train_labels = preprocess.flatten_multi_label('DataSet/semEval/semeval'+data+'-rawTextData',
																									'DataSet/semEval/semeval'+data+'-rawTextLabels',)

		test_data, test_labels = preprocess.flatten_multi_label('DataSet/semEval/semeval'+data+'_test-rawTextData',
																									'DataSet/semEval/semeval'+data+'_test-rawTextLabels',)

	print(len(train_labels + test_labels))
	encoded_labels, encoder, labeling = preprocess.encode_labels\
		(labels=train_labels + test_labels, scheme=subtask, one_hot=one_hot)

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

	train_labels = [list(compress(x, subtask)) for x in train_labels]
	test_labels =  [list(compress(i, subtask)) for i in test_labels]

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

		
def Hannah_pipe(train_cutoff = 8000, single=True):
	
	entity_dict, attr_dict = parse_hannah_legend('DataSet//organic//legend.csv')
	data, entities, attr = parse_hannah_csv('DataSet//organic//comments_PB3.csv')

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
		labels = zip(entities, attr)
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

	encoded_labels, encoder, labeling = preprocess.encode_labels(labels = labels, scheme=[True, True], one_hot=False )
	inverted_labels= encoded_labels# label_encoder.inverse_transform(encoded_labels)

	#sanity check
	if single:
		for i in range(len(data)):
			# print(data[i], labels[i], labeling[inverted_labels[i]])
			assert labeling[inverted_labels[i]] == labels[i]
	else:
		reoriginal=[]
		for i in inverted_labels:
			tmp=[]
			for e in i:
				tmp.append(labeling[e])
			reoriginal.append(tmp)
		for i in range(len(data)):
			for j in range(len(labels[i])):
				assert list(labels[i][j]) in reoriginal[i]

	try:
		with open("FeatureExtrac//hannah-openAI-data", 'rb') as fp:
			data_vecs = pickle.load(fp)
	except:
		data_vecs = openAI_transform(data, dmp_name="FeatureExtrac//hannah-openAI-data")
	
	print(len(data_vecs))
	test_vecs = data_vecs[train_cutoff:]
	train_vecs = data_vecs[:train_cutoff]
	encoded_test_labels = encoded_labels[train_cutoff:]
	encoded_train_labels=encoded_labels[:train_cutoff]
	test_data = data[train_cutoff:]
	train_data = data[:train_cutoff]
	train_labels = labels[:train_cutoff]
	test_labels = labels[:train_cutoff]

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


