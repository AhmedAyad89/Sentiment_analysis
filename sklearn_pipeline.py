import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier,RadiusNeighborsClassifier
from sklearn.neural_network import MLPClassifier,MLPRegressor
from sklearn.decomposition import PCA, FastICA, FactorAnalysis
from sklearn.cluster import DBSCAN,AffinityPropagation, KMeans
from sklearn.model_selection import train_test_split
from sklearn import metrics


def supervised(data, data_vecs, labels, test_vecs=None, test_labels=None):


	if(test_vecs is None):
		X_train, X_test, y_train, y_test = train_test_split(data_vecs, labels, test_size=0.2)
	else:
		X_train = data_vecs
		X_test= test_vecs
		y_train = labels
		y_test=test_labels

	# clf = LogisticRegression(penalty='l1')
	#clf = KNeighborsClassifier(weights='distance',n_neighbors=10, p=2, n_jobs=10)
	clf = MLPClassifier(hidden_layer_sizes=(512, 512), alpha=0.0002, batch_size=200)
	clf.fit(X_train, y_train)

	pred = clf.predict(X_test)

	# M = metrics.confusion_matrix(y_test, pred)
	#
	# for i,line in enumerate(M):
	# 	print(labeling[i])
	# 	print(line)
	# 	print('\n-----------------')

	try:
		print(metrics.accuracy_score(y_test, pred))
	except:
		print('x')
	try:
		print('micro', metrics.f1_score(y_test, pred, average='micro'))
		print('macro', metrics.f1_score(y_test, pred, average='macro'))
		print('weight', metrics.f1_score(y_test, pred, average='weighted'))
	except:
		print('x')
	print('samples', metrics.f1_score(y_test, pred, average='samples'))

	print('\n train errors\n-----------------------')
	pred = clf.predict(X_train)


	try:
		print(metrics.accuracy_score(y_train, pred))
	except:
		print('x')
	try:
		print('micro', metrics.f1_score(y_train, pred, average='micro'))
		print('macro', metrics.f1_score(y_train, pred, average='macro'))
		print('weight', metrics.f1_score(y_train, pred, average='weighted'))
	except:
		print('x')
	print('samples', metrics.f1_score(y_train, pred, average='samples'))


def unsupervised(data, data_vecs, labels):
	model = KMeans(n_clusters=15)
	labels = model.fit_predict(data_vecs)
	for i in range(len(labels)):
		print(labels[i], data[i])
		print('\n--------------------')
	# for i in model.cluster_centers_indices_:
	# 	print(data[i])
	# print(len(model.cluster_centers_indices_))

	for i in range(8):
		for counter,p in enumerate(labels):
			if (p==i):
				print(data[counter])

		print('\n\n\n=-------------------------------')

def decomp(data, data_vecs, labels):
	model = FactorAnalysis(n_components=3)
	reduced_data = model.fit_transform(data_vecs)

	for i,r in enumerate(reduced_data):
		print(r)
		print(data[i])
		print('\n-----------------------------------\n')