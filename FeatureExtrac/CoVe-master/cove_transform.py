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
		for s in content[:1000]:
			s = s.split(' ')
			if s[0] in vocab.keys():
				vocab[s[0]] = s[1:]
	if pickle_dmp is not None:
		with open(pickle_dmp, 'wb') as fp:
			pickle.dump(vocab, fp)
	return vocab

def sentence_to_glove(sentences, vocab):
	glove_sentences =[]
	for sentence in sentences:
		s = sentence.split(' ')
		tmp = []
		for word in s:
			tmp.append(vocab[word])
		glove_sentences.append(tmp)
	# for i in range(len(sentences)):
	# 	print(len(sentences[i]), len(glove_sentences[i]) )
	# 	print(sentences[i], glove_sentences[i])
	return glove_sentences

def cove_transform(glove_sentences, pickle_dmp=None):
	cove_model = load_model('Keras_CoVe.h5')
	encoded_sent = cove_model.predict(glove_sent)
	if pickle_dmp is not None:
		with open(pickle_dmp, 'wb') as fp:
			pickle.dump(vocab, fp)
	return encoded_sent

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
