# -*- coding: utf-8 -*-

import io
import numpy as np


class DataPreprocessor:
	"""
		This class preprocess the trainingsdata.
		It extracts the labels, removes unneeded chars and converts labels to one-hots.

		Attributes:
			charsToBeRemoved: string of chars, which will be ignored by the machine learning approach
			preprocessedData: the data, which was read without not needed chars and written to lowercase.
	"""
	charsToBeRemoved = "-?,.!\n"
	preprocessedData = []

	def __init__(self, number_of_classes):
		"""
		This method initializes the DataPreprocessor.
		:param number_of_classes: the number of categories, a sentence can be classified to
		"""
		self.number_of_classes = number_of_classes

	def preprocess_data(self, dataSrc):
		"""
			This method reads data out from the source, removes the punctuation and makes the data to lower-case.
		:param dataSrc: the source of the training-data
		"""
		with io.open(dataSrc, encoding="utf-8") as f:
			raw_data = f.readlines()
			for x in range(0, len(raw_data)):
				for ch in self.charsToBeRemoved:
					if (ch in raw_data[x]):
						raw_data[x] = raw_data[x].replace(ch, "")
				self.preprocessedData.append(raw_data[x].lower())

	@staticmethod
	def parse_labels(raw_data):
		"""
			This method parses the labels out of the raw-data.
		:param raw_data: the sentences, with the label
		:return: the labels
		"""
		y_labels = []
		for sentence in raw_data:
			clear_sentence = sentence.replace("|", "")
			try:
				y_labels.append(int(clear_sentence.split()[-1:][0]))
			except ValueError:
				print("Can not convert this, it has no label: ", sentence)
		return y_labels

	@staticmethod
	def parse_input_data(raw_data):
		"""
			This method parses the sentences out of the raw-data.
		:param raw_data: the sentences, with the label
		:return: the sentences, without a label
		"""
		x_inputs = []
		for sentence in raw_data:
			sentence = sentence.replace("|", "")
			sentence_no_digits = sentence.rsplit(' ', 1)[0]
			x_inputs.append(sentence_no_digits.split())
		return x_inputs

	@staticmethod
	def get_one_hot(label, number_of_classes):
		"""
			A function to convert numbers to one hot vectors
		:param label: the label, where the array should contain 1 instead of 0
		:param number_of_classes: the size of the one-hot array
		:return: the label, represented as one-hot-array
		"""
		return np.eye(number_of_classes)[np.array(label).reshape(-1)]

	def get_data(self, data_src):
		"""
			This function returns the preprocessed trainings-data in two different lists.
		:param data_src: the source of the trainings-data
		:return: the preprocessed trainings-data in two different lists
		"""
		self.preprocess_data(data_src)
		x = self.parse_input_data(self.preprocessedData)
		y = self.parse_labels(self.preprocessedData)
		y = self.get_one_hot(y, self.number_of_classes)
		return x, y