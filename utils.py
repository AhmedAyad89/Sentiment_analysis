import numpy as np
from sklearn.utils import shuffle as skshuffle

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





if __name__ == '__main__':
	d = [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16]
	l = ['a','b','c','d','a','b','c','d','a','b','c','d','a','b','c','d' ]
	D = data_wrap(d,l, batch_size=3)

	for i in range(15):
		print(D.next())
		print('\n---------')