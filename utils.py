import numpy as np
from sklearn.utils import shuffle as skshuffle
import json
import csv
from sklearn.preprocessing import scale
from sklearn import metrics
import codecs

def multi_hot_decode(list):
	L=[]
	for line in list:
		indices = [i for i, x in enumerate(line) if x == 1]
		L.append( indices )
		print(line,L)
	return L

class data_wrap(object):

	def __init__(self, data,labels= None,shuffle =True, batch_size=32 ):
		if(labels is None):
			np.random.shuffle(data)
			self.data = data
			self.labels=None

		else:
			self.labels = labels
			self.data = data
			self.data, self.labels = skshuffle(self.data, self.labels)

		self.data_dim = len(data[0])
		self.lenght_data = len(data)
		self.step_count=0
		self.batch_size = batch_size
		self.batch_index=0
		self.shuffle = shuffle


	def next(self):
		diff = self.batch_index + self.batch_size - self.lenght_data
		if(diff == self.batch_size):
			diff=0
			self.batch_index=0
		diff = max(diff,0)

		if(diff > 0):
			#print(self.data[self.batch_index: self.batch_index + self.batch_size])
			x = np.concatenate((self.data[self.batch_index: self.batch_index + self.batch_size], \
													self.data[0: diff]))
			if (self.labels is not None):
				y = np.concatenate((self.labels[self.batch_index: self.batch_index + self.batch_size], \
														self.labels[0:diff]))

			self.batch_index=diff
			if (self.shuffle):
				if (self.labels is None):
					np.random.shuffle(self.data)
				else:
					self.data, self.labels = skshuffle(self.data, self.labels)
		else:
			x = self.data[self.batch_index: self.batch_index + self.batch_size]
			if (self.labels is not None):
				y = self.labels[self.batch_index: self.batch_index + self.batch_size]

			self.batch_index+=self.batch_size
		if(self.labels is not None):
			return x,y
		return x

def single_label_metrics(pred, labels, encoded_labels, labeling, data):
	try:
		print('micro', metrics.f1_score(encoded_labels, pred, average='micro'))
	# print('macro', metrics.f1_score(l, pred, average='macro'))
	# print('weight', metrics.f1_score(l, pred, average='weighted'))
	except Exception as e:
		print(e)
	print('f1 samples', metrics.f1_score(encoded_labels, pred, average='samples'))
	print('precision samples', metrics.precision_score(encoded_labels, pred, average='samples'))
	print('recall samples', metrics.recall_score(encoded_labels, pred, average='samples'))
	decoded_pred = np.where(pred == 1)[1]
	raw_pred=[]
	for i in range(len(labels)):
		if labels[i] == ['not relevant', 'not relevant']:
			continue
		raw_pred.append(labeling[decoded_pred[i]])
		print(i, '- ', data[i])
		print('Prediction: ', (labeling[decoded_pred[i]]), decoded_pred[i])
		print('Label: ', labels[i])
		print('\n', '\n--------------------------\n')
	return raw_pred

def multi_label_metrics(pred, labels, encoded_labels, labeling, data, mute=True):
	try:
		print('micro', metrics.f1_score(encoded_labels, pred, average='micro'))
		# print('f1 samples', metrics.f1_score(encoded_labels, pred, average='samples'))
		# print('precision samples', metrics.precision_score(encoded_labels, pred, average='samples'))
		# print('recall samples', metrics.recall_score(encoded_labels, pred, average='samples'))
		print('accuracy', metrics.accuracy_score(encoded_labels, pred))
	except Exception as e:
		print('accuracy', metrics.accuracy_score(encoded_labels, pred))
		print(e)

	decoded_pred = []
	for p in pred:
		decoded_pred.append(np.where(p == 1)[0])
	if not mute:
		for i in range(len(labels)):
			try:
				if (not labels[i]) and (not decoded_pred[i]):
					continue
			except:
				p=4
			print(i, '- ', data[i])
			print('Prediction: ',  [labeling[x] for x in decoded_pred[i]] )
			print('Label: ', labels[i])
			print('\n', '\n--------------------------\n')

def print_raw_predictions(pred, labels, encoded_labels, labeling, data, mute=True):
	decoded_pred = []
	for p in pred:
		decoded_pred.append(np.where(p == 1)[0])
	if not mute:
		for i in range(len(labels)):
			print(i, '- ', data[i])
			print('Prediction prob: ', [(labeling[x],pred[i][x]) for x in range(len(pred[i]))])
			print('Label: ', labels[i])
			print('\n', '\n--------------------------\n')


#gets tokenized comment texts
def parse_tokenized_comments_json(path, sentence_level = True):
	data = json.load(open(path))
	comments=[]
	sentences = []
	count=0
	for line in data:
		for comment in line['comments']:
			if(comment['comment_text_tokenized'] != []):
				comments.append(comment['comment_text_tokenized'])
	if(not sentence_level):
		return comments
	for c in comments:
		sentence = []
		for i in c:
			sentence.append(' '.join(i))
		sentences.append(' '.join(sentence))
	return sentences

# gets tokenized comment texts
def parse_tokenized_for_labelling(path, sentence_level=True):
	data = json.load(open(path))
	comments = []
	sentences = []
	count = 0
	for line in data:
		for comment in line['comments']:
			if (len(comment['comment_text']) > 60):
				comments.append(comment['comment_text_tokenized'])
	if (not sentence_level):
		return comments
	unflatenned=[]
	print(len(comments))
	sel = np.random.choice(comments, 2000)
	for c in sel:
		sentence = []
		for i in c:
			sentence.append(' '.join(i))
		unflatenned.append(sentence)
		sentences.append(' '.join(sentence))
	return unflatenned

def parse_hannah_legend(path):
	with open(path, 'r', encoding='mbcs') as fp:
		reader = csv.reader(fp, delimiter=',')
		listed = [x for x in reader]
	entity_dict = {}
	attr_dict = {}
	for i, line in enumerate(listed):
		if line[0] == 'entity':
			entity_dict[line[1]] = line[2]
		if line[0] == 'attribute':
			attr_dict[line[1]] = line[2]
	entity_dict['NA'] = 'not relevant'
	attr_dict['NA'] = 'not relevant'
	entity_dict['rel'] = 'relevant'
	attr_dict['rel'] = 'relevant'
	return entity_dict, attr_dict

def parse_hannah_csv(path):
	data = []
	attr = []
	entities = []
	num = []
	relevance = []
	sentiments = []
	with open(path,'r', encoding='mbcs') as fp:
		reader = csv.reader(fp, delimiter=',')

		listed = [x for x in reader ]

		counter=0
		for i,line in enumerate(listed):
			if line[0] == '***********':
				continue
			if not line[6]:
				continue
			num.append(line[0])
			data.append(line[6])
			attr.append([line[5]])
			entities.append([line[4]])
			sentiments.append([line[3]])
			relevance.append(line[2])
			j = i
			while (not listed[j + 1][0]):
				j += 1
				#print(j)
				entities[counter].append(listed[j][4])
				attr[counter].append(listed[j][5])
				sentiments[counter].append(listed[j][3])
			counter += 1


	print(len(data))

	for i in range(len(data)):
		if relevance [i] == '9' or relevance [i] == 'g':
			if entities[i] == ['']:
				assert attr[i] == ['']
				entities[i] = ['rel']
				attr[i] = ['rel']
		else:
			entities[i] = ['NA']
			attr[i] = ['NA']
		assert len(entities[i]) == len(attr[i])
		#print(data[i], relevance[i], entities[i], attr[i])

	return data, entities, attr, sentiments

def normalize_feat(features):
	features = np.asarray(features)
	features = scale(features)
	return features

if __name__ == '__main__':
	# comments = parse_tokenized_for_labelling("DataSet/organic/tokenized/en.json")
	#
	# rows=[]
	# for comment_num,comment in enumerate(comments):
	# 	row=[]
	# 	for sentence_num, sentence in enumerate(comment):
	# 		#row=','.join([str(comment_num), str(sentence_num),'', '', sentence])
	# 		row=[str(comment_num), str(sentence_num),'', '','','', sentence]
	# 		rows.append(row)
	# 	rows.append(['***********'])
	# with open('comments2.csv', 'w', encoding='utf-8-sig') as fp:
	# 	fieldnames = ['Comment_number', 'Sentence_number', 'Can\'t tell', 'Sentiment', 'Entity', 'Attribute', 'Sentence' ]
	# 	writer = csv.writer(fp, delimiter=',')
	# 	writer.writerow(fieldnames)
	# 	writer.writerows(rows)


	entity_dict, attr_dict = parse_hannah_legend('DataSet//organic//legend.csv')
	print(entity_dict)
	print(attr_dict)
	data, entities, attr, sent = parse_hannah_csv('DataSet/organic/19-03-05_comments_HD.csv')




	for i in range(6000):
		for j,e in enumerate(entities[i]):
				print(data[i], entity_dict[e], attr_dict[attr[i][j]], sent[i][j])