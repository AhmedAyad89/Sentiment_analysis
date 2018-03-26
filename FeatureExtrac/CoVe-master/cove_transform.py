from DataSet.dataset_pipelines import *
import numpy as np
import pickle
from keras.models import load_model
def build_vocab(text, pickle_dmp=None):
	vocab = {}
	for sentence in text:
		sentence = sentence.split(' ')
		for word in sentence:
			vocab[word] = np.zeros(shape=300) + 0.00000000001

	with open('glove.840B.300d.txt', 'r', encoding='utf-8') as fp:
		content = fp.readlines()
		for s in content:
			s = s.split(' ')
			if s[0] in vocab.keys():
				vocab[s[0]] = s[1:]
	if pickle_dmp is not None:
		with open(pickle_dmp, 'wb') as fp:
			pickle.dump(vocab, fp)
	return vocab

def sentence_to_glove(sentences, vocab, dmp=None):
	len_embedding =300
	max_len = 100
	glove_sentences = np.zeros(shape=[len(sentences), max_len, len_embedding], dtype=np.float64) + 0.0000000001
	for i, sentence in enumerate(sentences):
		s = sentence.split(' ')
		for j, word in enumerate(s):
			if j > max_len:
				print('broke')
				break
			glove_sentences[i][j][:] = np.asarray(vocab[word])

	# for i in range(len(sentences)):
	# 	print(len(sentences[i]), len(glove_sentences[i]) )
	# 	print(sentences[i], glove_sentences[i])

	if dmp is not None:
		with open(dmp, 'wb') as fp:
			pickle.dump(glove_sentences, fp)

	return glove_sentences

def cove_transform(glove_sentences, pickle_dmp=None):
	cove_model = load_model('C:\\Users\\Ayad\\Desktop\\Academic\\NLP\\sA\\Sentiment-analysis-code\\FeatureExtrac\\CoVe-master\\Keras_CoVe.h5')
	encoded_sent = cove_model.predict(glove_sentences)
	if pickle_dmp is not None:
		with open(pickle_dmp, 'wb') as fp:
			pickle.dump(encoded_sent, fp)
	return encoded_sent

def cove_pipeline(data, glove_dmp=None, cove_dmp=None):
	vocab = build_vocab(data)
	glove_sentences = sentence_to_glove(data, vocab, glove_dmp)
	cove_sentences = cove_transform(glove_sentences, cove_dmp)
	return glove_sentences, cove_sentences

def build_embedding_matrices(text, pickle_dmp=None):
	vocab_index = {}
	idx = 1
	vocab_index[0] = 0
	for sentence in text:
		sentence = sentence.split(' ')
		for word in sentence:
			if word not in vocab_index.keys():
				vocab_index[word] = idx
				idx+=1
	vocab =  np.zeros(shape=(idx ,300))
	vocab[0]= np.zeros(300) - 0.000000001
	with open('glove.840B.300d.txt', 'r', encoding='utf-8') as fp:
		content = fp.readlines()
		for s in content:
			s = s.split(' ')
			if s[0] in vocab_index.keys():
				vocab[vocab_index[s[0]]] = s[1:]

	if pickle_dmp is not None:
		with open(pickle_dmp, 'wb') as fp:
			pickle.dump(vocab, fp)
	return vocab, vocab_index

#convert a sentences to indices
def convert_to_indices(text, max_len, vocab_index):
	sentence_matrix = np.zeros((len(text), max_len))
	for i, sentence in enumerate(text):
		s = sentence.split(' ')
		for j, word in enumerate(s):
			if j >= max_len:
				break
			sentence_matrix[i][j] = vocab_index[word]
	return sentence_matrix

if __name__ == '__main__':
	sentences = ['I love Bananas', 'I fucking hate semeval']
	vocab = build_vocab(sentences)
	glove_sent = sentence_to_glove(sentences, vocab)
	glove_sent = np.asarray(glove_sent)
	print(glove_sent.shape)
	cove_model = load_model('Keras_CoVe.h5')
	encoded_sent = cove_model.predict(glove_sent)
	print(encoded_sent)
	print(len(encoded_sent[0]))
