import tensorflow as tf
from keras import activations
from keras.models import Sequential, Model
from keras.layers import Dense, Conv1D, MaxPooling1D, Input, Embedding, Flatten, concatenate, BatchNormalization, Dropout
import keras



def keras_conv_net(vocab_size, embd_matrix):
	print(vocab_size)
	cut=35
	cove_input = Input((cut, 600))
	text_input = Input((cut, ))
	openAI_input= Input((4096,))
	embd = Embedding(input_dim =vocab_size, output_dim=300, weights=[embd_matrix], input_length=cut)(text_input)
	merged_1 =concatenate([embd,cove_input])
	merged_1 = Dropout(0.5)(merged_1)
	conv_1 = Conv1D(filters=128, kernel_size=6, activation='relu',
									kernel_regularizer=keras.regularizers.l2(0.00005), input_shape=(cut, 900))(merged_1)
	pool_1=MaxPooling1D(pool_size=10)(conv_1)
	flattened_1 = Flatten()(pool_1)
	merged_2 = concatenate([flattened_1, openAI_input])
	normalized_1 = BatchNormalization()(merged_2)
	dense_1 = Dense(units=512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(normalized_1)
	normalized_2 = BatchNormalization()(dense_1)
	dense_2 = Dense(units=512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(dense_1)
	out = Dense(units=88, activation='softmax')(dense_2)
	out = Dense(units=88, activation='sigmoid')(out)
	model = Model(inputs=[text_input, cove_input, openAI_input], outputs=out)
	model.compile(optimizer='adam',
								loss='binary_crossentropy')

	print(model.summary())
	return model

