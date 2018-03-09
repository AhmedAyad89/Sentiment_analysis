import pickle
import operator
from sklearn.cluster import KMeans
import spacy as sc
import numpy as np

def unsupervised(data, data_vecs,num_clusters = 10):
		model = KMeans(n_clusters=num_clusters)
		labels = model.fit_predict(data_vecs)
		for i in range(num_clusters):
				count =0
				for counter,p in enumerate(labels):
						if (p==i):
								count+=1
								print(data[counter])
						if(count>50):
								break
				print('\n\n\n=-------------------------------',i)

def bow_clusters_features(dict_path='C:\\Users\\Ayad\\Desktop\\Academic\\NLP\\sA\\Sentiment-analysis-code\\cluster_dicts', data =None):
	with open(dict_path, 'rb') as fp:
		dict = pickle.load(fp)
	lists = []
	for c in dict:
		list = []
		for w in c.keys():
			list.append(str(w))
		lists.append(list)

	features = []
	for sentence in data:
		tok = sentence.split(' ')
		feature = np.zeros(len(dict))
		for word in tok :
			for i, cluster in enumerate(lists):
				if(word in cluster):
					feature[i] = 1
		features.append(feature)

	assert len(features) == len(data)
	return features

if __name__ == '__main__':
	data = ['I eat health', 'I hate monsanto crop apples']
	features = bow_clusters_features('C:\\Users\\Ayad\\Desktop\\Academic\\NLP\\sA\\Sentiment-analysis-code\\cluster_dicts', data)

	for i in range(2):
		print(data[i], features[i])