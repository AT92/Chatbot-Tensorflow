# -*- coding: utf-8 -*-


import numpy as np

import tflearn
from tflearn import input_data, dropout, fully_connected, conv_2d, max_pool_2d, regression, DNN

import tensorflow as tf


class Classifier:
	"""
		This class crates an convolutional neural network and trains it to create a model, for
		predicting the class of the question.

		Attributes:
			network:                the convolutional neural network
			model:                  the model, which has learned how to predict the questions
			max_sentence_length:    the maximum amount of words in a sentence
			embedding_dim:          the dimension of the vector each word is embedded in
			number_of classes:      the amount of categories, a sentence can be classified in
			learnign_rate:          this parameter influences, how big is the optimisation step to the right direction of a model
									during the learning process. If the learning_rate is to low, it last to long, for the model to learn,
									if the step is to big, the model can not be optimized really, because it is difficult to hit the right
									value.
			test_data_proportion:   the preoportion, how much of the training-data should be used for the model validation and not for the
									training itself.
			epochs:                 the amount of times, the modell should be learned
	"""
	network = None
	model = None

	def __init__(self, config):
		"""
			This method initializes all the parameter got from the configuration object.
		:param config: the configuration of the model
		"""
		self.max_sentence_length = config.parameters["networks"][1]["max_sentence_length"]
		self.embedding_dim = config.parameters["networks"][0]["embedding_dim"]
		self.number_of_classes = config.parameters["networks"][1]["number_of_classes"]
		self.learning_rate = config.parameters["networks"][1]["learning_rate"]
		self.test_data_proportion = config.parameters["networks"][1]["test_data_proportion"]
		self.epochs = config.parameters["networks"][1]["epochs"]

	def save_model(self):
		"""
			This method saves the created model to the drive
		"""
		self.model.save("./classifier/models/classifier.tflearn")

	# this method loads a created model
	def load_model(self):
		"""
			This method loads a created model from the drive.
		"""
		model = None
		with tf.Graph().as_default():
			self.create_network()
			model = DNN(self.network, tensorboard_verbose=0)
			model.load("./classifier/models/classifier.tflearn")
			self.model = model


	def create_network(self):
		"""
			This method creates the convolutional neural network.
		"""

		# creating the input layer
		network = input_data(shape=[None, self.max_sentence_length, self.embedding_dim, 1], name='input')
		# creating the first convolutional layer
		network = conv_2d(network, 32, 3, activation='relu', regularizer="L2")
		# creating the first pool layer
		network = max_pool_2d(network, 2)
		# creating the second convolutional layer
		network = conv_2d(network, 64, 5, activation='relu', regularizer="L2")
		# creating the first pool layer
		network = max_pool_2d(network, 2)
		# droping out some tensors, for avoiding the network to overfit
		network = dropout(network, 0.5)
		# creating the soft-max layer
		network = fully_connected(network, self.number_of_classes, activation='softmax')
		self.network = regression(network, optimizer='adam', learning_rate=self.learning_rate, loss='categorical_crossentropy', name='target')


	# this method trains the model with the passed training data
	def train_model(self, x_train, y_train):
		x_train = np.array(x_train, np.float32)
		x_train = np.array([i for i in x_train]).reshape(-1, self.max_sentence_length, self.embedding_dim, 1)

		# Training
		self.model = tflearn.DNN(self.network, tensorboard_verbose=0)
		self.model.fit(x_train, y_train, validation_set=self.test_data_proportion, n_epoch=self.epochs, snapshot_step=500, show_metric=True, run_id='CNN')

	# this function return the predicted values, which the model has predicted for the input as a tensor
	def predict(self, tensor):
		return self.model.predict(tensor)