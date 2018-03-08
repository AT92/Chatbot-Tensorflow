# -*- coding: utf-8 -*-

import pickle

import numpy as np
import tensorflow as tf
from .DataPreprocessor import DataPreprocessor
from .ModelConfiguration import ModelConfiguration
from .WordEmbedding import WordEmbedding

from classifier.Classifier import Classifier


class Model:
	"""
		This class is used for creating and using the sentence classification model.

		For creating a classification model, firs a word-embedding model shoould be created or loaded,
		in the case if one already exists.

		Attributes:
			word2int:               a dictionary, which stores for each word an unique id as an integer
			int2word:               the reversed word2int dictionary
			vectors:                an ordered set of words embedded in a vector with the size of embedding_dim
			max_sentence_length:    the maximum amount of words in a sentence
			PADDING:                the constant, with which a sentence is padded, if it has less than max-sentence-length words.
									Besides that it is used for unknown words, which does not have any embedding.
			x_training:             the extracted sentences from raw-data
			y_training:             the extracted labels from raw-data
			cl_model:               the classification model
			vocab_size:             the vocabulary_size of the model

	"""
	word2int = {}
	int2word = {}
	vectors = []
	max_sentence_length = None
	PADDING = None
	x_training = None
	y_training = None
	cl_model = None
	vocab_size = None
	classifier = None

	def __init__(self):
		self.config = ModelConfiguration()
		config = ModelConfiguration()
		self.max_sentence_length = config.parameters["networks"][1]["max_sentence_length"]
		self.empedding_dim = config.parameters["networks"][0]["embedding_dim"]
		self.PADDING = np.zeros(self.empedding_dim, dtype=np.dtype(np.float32))

	def load_training_data(self):
		preprocessor = DataPreprocessor(self.config.parameters["networks"][1]["number_of_classes"])
		self.x_training, self.y_training = preprocessor.get_data(self.config.parameters["training_data_src"])

	def create_word_embedding(self):
		self.load_training_data()
		word_embedding = WordEmbedding(sentences=self.x_training, config=self.config.get_word_embedding_config())
		self.vectors = word_embedding.vectors
		self.vocab_size = word_embedding.vocab_size
		self.config.parameters['networks'][0]['vocab_size'] = word_embedding.vocab_size
		self.config.save()
		self.word2int = self.load_obj("word2int")
		self.int2word = self.load_obj("int2word")

	def load_word_embeddings(self):
		tf.reset_default_graph()
		w1 = tf.get_variable("weights_first_layer", shape=[self.config.parameters["networks"][0]["vocab_size"], self.empedding_dim])
		b1 = tf.get_variable("biases_first_layer", shape=[self.empedding_dim])  # bias

		saver = tf.train.Saver()

		self.word2int = self.load_obj("word2int")
		self.int2word = self.load_obj("int2word")
		with tf.Session() as sess:
			# Restore variables from disk.
			saver.restore(sess, "./classifier/models/w2v.ckpt")
			self.vectors = sess.run(w1 + b1)

	def create_model(self):
		self.load_training_data()
		classifier = Classifier(self.config)
		classifier.create_network()
		classifier.train_model(self.sentences_2_tensor(self.x_training), self.y_training)
		classifier.save_model()
		self.classifier = classifier.load_model()

	def predict(self, sentence):
		if (self.classifier is None):
			self.classifier = Classifier(self.config)
			self.classifier.load_model()

		sentence_as_tensor = self.sentence_2_tensor(sentence)
		sentence_as_tensor = np.array(sentence_as_tensor)
		sentence_as_tensor = sentence_as_tensor.reshape(1, self.max_sentence_length, self.empedding_dim, 1)
		classification = self.classifier.predict(sentence_as_tensor)[0]
		if (np.amax(classification) < 0.8):
			return -1
		return np.argmax(classification)
	
	def sentences_2_tensor(self, sentences):
		tensors = []
		for sentence in sentences:
			tensor = []
			for word in sentence:
				tensor.append(self.vectors[self.word2int[word]])
			while len(tensor) < self.max_sentence_length:
				tensor.append(self.PADDING)
			tensors.append(tensor)
		return tensors

	def sentence_2_tensor(self, sentence):
		tensor = []
		sentence = sentence.split()
		for word in sentence:
			try:
				tensor.append(self.vectors[self.word2int[word]])
			except Exception:
				tensor.append(self.PADDING)
		while len(tensor) < self.max_sentence_length:
			tensor.append(self.PADDING)
		return tensor

	@staticmethod
	def load_obj(name):
		with open('./classifier/dictionary/' + name + '.pkl', 'rb') as f:
			return pickle.load(f)