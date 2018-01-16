from time import time

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter

from sklearn import manifold, datasets

# Next line to silence pyflakes. This import is needed.
Axes3D

def manifold_viz(data_vecs, data_labels = None, color = None):
	n_points = 2500
	#X, color = datasets.samples_generator.make_s_curve(n_points, random_state=0)
	X = data_vecs
	n_neighbors = 3
	n_components = 3

	fig = plt.figure(figsize=(20, 14))
	plt.suptitle("Manifold Learning with %i points, %i neighbors"
							 % (1000, n_neighbors), fontsize=14)


	ax = fig.add_subplot(251, projection='3d')
	if(color is not None):
		ax.scatter(X[:, 0], X[:, 1], X[:, 2],c = color, cmap=plt.cm.Spectral)
	else:
		ax.scatter(X[:, 0], X[:, 1], X[:, 2],  cmap=plt.cm.Spectral)
	if(data_labels is not None):
		for i, txt in enumerate(data_labels):
			ax.annotate(txt, (X[:, 0], X[:, 1]))
	ax.view_init(4, -72)

	methods = ['standard']
	labels = ['LLE']

	for i, method in enumerate(methods):
			t0 = time()
			Y = manifold.LocallyLinearEmbedding(n_neighbors, n_components,
																					eigen_solver='auto',
																					method=method).fit_transform(X)
			t1 = time()
			print("%s: %.2g sec" % (methods[i], t1 - t0))

			ax = fig.add_subplot(252 + i, projection='3d')
			plt.scatter(Y[:, 0], Y[:, 1],Y[:, 2], c=color, cmap=plt.cm.Spectral)
			if (data_labels is not None):
				for j, txt in enumerate(data_labels):
					# ax.annotate(txt, (Y[j, 0], Y[j, 1], Y[j, 2]))
					ax.text(Y[j, 0], Y[j, 1], Y[j, 2], txt)
			plt.title("%s (%.2g sec)" % (labels[i], t1 - t0))
			ax.xaxis.set_major_formatter(NullFormatter())
			ax.yaxis.set_major_formatter(NullFormatter())
			plt.axis('tight')
			break

	t0 = time()
	Y = manifold.Isomap(n_neighbors, n_components).fit_transform(X)
	t1 = time()
	print("Isomap: %.2g sec" % (t1 - t0))
	ax = fig.add_subplot(254, projection='3d')
	plt.scatter(Y[:, 0], Y[:, 1], Y[:, 2], c=color, cmap=plt.cm.Spectral)
	if(data_labels is not None):
		for i, txt in enumerate(data_labels):
			#ax.annotate(txt, (Y[i, 0], Y[i, 1], Y[i, 2]))
			ax.text(Y[j, 0], Y[j, 1], Y[j, 2], txt)

	plt.title("Isomap (%.2g sec)" % (t1 - t0))
	ax.xaxis.set_major_formatter(NullFormatter())
	ax.yaxis.set_major_formatter(NullFormatter())
	plt.axis('tight')


	t0 = time()
	mds = manifold.MDS(n_components, max_iter=100, n_init=1)
	Y = mds.fit_transform(X)
	t1 = time()
	print("MDS: %.2g sec" % (t1 - t0))
	ax = fig.add_subplot(255, projection='3d')
	plt.scatter(Y[:, 0], Y[:, 1], Y[:, 2], c=color, cmap=plt.cm.Spectral)
	if(data_labels is not None):
		for i, txt in enumerate(data_labels):
			# ax.annotate(txt, (Y[i, 0], Y[i, 1], Y[i, 2]))
			ax.text(Y[j, 0], Y[j, 1], Y[j, 2], txt)

	plt.title("MDS (%.2g sec)" % (t1 - t0))
	ax.xaxis.set_major_formatter(NullFormatter())
	ax.yaxis.set_major_formatter(NullFormatter())
	plt.axis('tight')


	t0 = time()
	se = manifold.SpectralEmbedding(n_components=n_components,
																	n_neighbors=n_neighbors)
	Y = se.fit_transform(X)
	t1 = time()
	print("SpectralEmbedding: %.2g sec" % (t1 - t0))
	ax = fig.add_subplot(256, projection='3d')
	plt.scatter(Y[:, 0], Y[:, 1], Y[:, 2], c=color, cmap=plt.cm.Spectral)
	if(data_labels is not None):
		for i, txt in enumerate(data_labels):
			# ax.annotate(txt, (Y[i, 0], Y[i, 1], Y[i, 2]))
			ax.text(Y[j, 0], Y[j, 1], Y[j, 2], txt)

	plt.title("SpectralEmbedding (%.2g sec)" % (t1 - t0))
	ax.xaxis.set_major_formatter(NullFormatter())
	ax.yaxis.set_major_formatter(NullFormatter())
	plt.axis('tight')

	t0 = time()
	tsne = manifold.TSNE(n_components=n_components, init='pca', random_state=0)
	Y = tsne.fit_transform(X)
	t1 = time()
	print("t-SNE: %.2g sec" % (t1 - t0))
	ax = fig.add_subplot(2, 5, 10, projection='3d')
	plt.scatter(Y[:, 0], Y[:, 1],Y[:,2], c=color, cmap=plt.cm.Spectral)
	if(data_labels is not None):
		for i, txt in enumerate(data_labels):
			# ax.annotate(txt, (Y[i, 0], Y[i, 1], Y[i, 2]))
			ax.text(Y[j, 0], Y[j, 1], Y[j, 2], txt)

	plt.title("t-SNE (%.2g sec)" % (t1 - t0))
	ax.xaxis.set_major_formatter(NullFormatter())
	ax.yaxis.set_major_formatter(NullFormatter())
	plt.axis('tight')

	plt.show()