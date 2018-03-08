# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import pickle


class WordEmbedding:
	"""
		This class creates the model for word-embedding
		based on the word2vec algorithm and skip-gram.

		It is used for representing the words as vectors,
		regarding their semantical meaning.

		Attributes:
			word2int:       a dictionary, which stores for each word an unique id as an integer
			int2word:       the reversed word2int dictionary
			sentences:      a list whith all sentences, the model will be trained on
			embedding_dim:  the dimension of a vector, which should represent a word
			window_size:    the size of the window used by the skip gram algorithm,
							so if there is an word with index i w(i), the algorithm will consider
							all words between w(i) and w(i + window_size) and w(i - wondow_size)
			iterations:     the amount of training steps for the word2vec algorithm
			batch_size:     the size of the batch, tensorflow should take for learning in one step.
							if the batch size is small, the process does not need much ram, but is slow
			learning_rate:  this parameter influences, how big is the optimisation step to the right direction of a model
							during the learning process. If the learning_rate is to low, it last to long, for the model to learn,
							if the step is to big, the model can not be optimized really, because it is difficult to hit the right
							value.
			words:          an ordered set of words, which should be embedded
			vectors:        an ordered set of words embedded in a vector with the size of embedding_dim
	"""

	word2int = {}
	int2word = {}

	def __init__(self, sentences, config):
		"""
			This method creates the new word-embedding model and trains it.
			First there will be created the dictionaries word2int and int2word, where
			each unique word gets an unique integer id. After that, the skip gram algorithm
			gets for each word the target-words, which are located in the window of the current word.
			After that a neural network with just one hidden layer will be created and learned.
			The weights + biases of this hidden layer represents the word embeddings. After all the model will
			be saved to the drive.

		:param sentences: the sentences, out of which the words should be embedded
		:param config: the configuration of the word embedding model.
		"""

		self.sentences = sentences
		self.embedding_dim = config["embedding_dim"]
		self.window_size = config["window_size"]
		self.iterations = config["iterations"]
		self.batch_size = config["batch_size"]
		self.learning_rate = config["learning_rate"]

		words = []

		#getting each word out of all sentences
		for sentence in sentences:
			for word in sentence:
				words.append(word)

		#sorting words and removing duplicates
		self.words = set(words)
		words = sorted(self.words)

		#getting the amount of words
		self.vocab_size = len(words)

		#creating a dictionary, so each word has an unique id
		for i, word in enumerate(words):
			self.word2int[word] = i
			self.int2word[i] = word

		#saving the dictionaries to the drive
		self.save_obj(self.word2int, "word2int")
		self.save_obj(self.int2word, "int2word")

		data = []


		# creating data for the skip gram algorithmus
		for sentence in sentences:
			for word_index, word in enumerate(sentence):
				for nb_word in sentence[
					max(word_index - self.window_size, 0): min(word_index + self.window_size, len(sentence)) + 1]:
					if nb_word != word:
						data.append([word, nb_word])

		x_train = []  # input word
		y_train = []  # output word

		# converting trainingsdata to one hots
		for data_word in data:
			x_train.append(self.to_one_hot(self.word2int[data_word[0]], self.vocab_size))
			y_train.append(self.to_one_hot(self.word2int[data_word[1]], self.vocab_size))

		# convert them to numpy arrays
		x_train = np.asarray(x_train)
		y_train = np.asarray(y_train)

		#creating 1-hiddenlayes-network

		x = tf.placeholder(tf.float32, shape=(None, self.vocab_size))
		y_label = tf.placeholder(tf.float32, shape=(None, self.vocab_size))

		w1 = tf.Variable(tf.random_normal([self.vocab_size, self.embedding_dim]), name="weights_first_layer")
		b1 = tf.Variable(tf.random_normal([self.embedding_dim]), name="biases_first_layer")  # bias
		hidden_representation = tf.add(tf.matmul(x, w1), b1)

		w2 = tf.Variable(tf.random_normal([self.embedding_dim, self.vocab_size]), name="weights_second_layer")
		b2 = tf.Variable(tf.random_normal([self.vocab_size]), name="biases_second_layer")
		prediction = tf.nn.softmax(tf.add(tf.matmul(hidden_representation, w2), b2))

		sess = tf.Session()
		init = tf.global_variables_initializer()
		sess.run(init)  # make sure you do this!
		# define the loss function:
		cross_entropy_loss = tf.reduce_mean(-tf.reduce_sum(y_label * tf.log(prediction), reduction_indices=[1]))
		# define the training_data step:
		train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(cross_entropy_loss)

		#defining the saver, to save the word-embedding modell
		saver = tf.train.Saver()

		#defining a period of training
		period = round((len(x_train) / self.batch_size))

		# starting the training
		print("Word-Embedding started")
		for _ in range(self.iterations):
			for i in range(period):
				batch_x = x_train[i * self.batch_size: (i + 1) * self.batch_size]
				batch_y = y_train[i * self.batch_size: (i + 1) * self.batch_size]
				sess.run(train_step, feed_dict={x: batch_x, y_label: batch_y})
			if (_ % 2 == 0):
				print("Word-embedding step: ", _)


		saver.save(sess, "./classifier/models/w2v.ckpt")
		self.vectors = sess.run(w1 + b1)


	@staticmethod
	def to_one_hot(word_id, vocab_size):
		"""
			A function to convert numbers to one hot vectors
		:param dword_id: index, where the array should contain 1 instead of 0
		:param vocab_size: the size of the one-hot array
		:return: a word, represented as one-hot-array
		"""
		temp = np.zeros(vocab_size)
		temp[word_id] = 1
		return temp

	@staticmethod
	def save_obj(obj, name):
		"""
			A function for saving word dictionaries, which contains word - id pairs
		:param obj:  the object, which should be stored
		:param name: the name, of the stored object.
		:return:
		"""
		with open('./classifier/dictionary/' + name + '.pkl', 'wb') as f:
			pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)