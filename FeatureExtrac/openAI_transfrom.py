from depend.openAI.encoder import Model
import numpy as np
import pickle



def openAI_transform(data, dmp_name=None):

	model = Model()
	text_features = model.transform(data)

	if(dmp_name is not None):
		with open(dmp_name, 'wb') as fp:
				pickle.dump(text_features, fp)

	return text_features


if __name__ == '__main__':
	print('x')




