# -*- coding: utf-8 -*-

import tensorflow as tf
import io
import numpy as np

import tflearn

from tflearn import input_data, dropout, fully_connected, conv_2d, max_pool_2d, regression, local_response_normalization


def get_one_hot(targets, nb_classes):
    return np.eye(nb_classes)[np.array(targets).reshape(-1)]



word2int = {}
int2word = {}
words = []





cleared_sentences = []


EMBEDDING_DIM = 5
MAX_SENTENCE_LENGTH = 12

# Training Parameters
learning_rate = 0.5
num_iters = 200
batch_size = 240
display_step = 10






x_train = []
y_train = []


with io.open('../training_data/input_data.txt', encoding="utf-8") as f:
	raw_sentences = f.readlines()

for sentence in raw_sentences:
	clear_sentence = sentence.replace("|", "")
	try:
		y_train.append(int(clear_sentence[len(clear_sentence) - 2]))
	except ValueError:
		print("Can not convert this: ", sentence)
	clear_sentence = ''.join([i for i in clear_sentence if not i.isdigit()])
	cleared_sentences.append(clear_sentence)
	for word in sentence.split():
		if (not word.isdigit() and word != "|"):
			words.append(word)


y_train = get_one_hot(y_train, 5)


words = set(words)
words = sorted(words)
vocab_size = len(words)

for i, word in enumerate(words):
	word2int[word] = i
	int2word[i] = word

tf.reset_default_graph()
W1 = tf.get_variable("weights_first_layer", shape=[vocab_size, 5])
b1 = tf.get_variable("biases_first_layer", shape=[5])  # bias

saver = tf.train.Saver()
# vectors = []

with tf.Session() as sess:
	# Restore variables from disk.
	saver.restore(sess, "../models/w2v.ckpt")
	vectors = sess.run(W1 + b1)





padding = np.zeros(EMBEDDING_DIM, dtype=np.dtype(np.float32))


for sentence in cleared_sentences:
	emb_sentence = []
	for word in sentence.split():
		emb_sentence.append(vectors[word2int[word]])

	while len(emb_sentence) < 12:
		emb_sentence.append(padding)

	x_train.append(emb_sentence)




x_train = np.array(x_train, np.float32)
print(x_train.shape)


# Building convolutional network
network = input_data(shape=[None, 12, 5, 1], name='input')
network = conv_2d(network, 32, 3, activation='relu', regularizer="L2")
network = max_pool_2d(network, 2)
network = conv_2d(network, 64, 5, activation='relu', regularizer="L2")
network = max_pool_2d(network, 2)
network = dropout(network, 0.5)
network = fully_connected(network, 5, activation='softmax')
network = regression(network, optimizer='adam', learning_rate=0.01, loss='categorical_crossentropy', name='target')




x = np.array([i for i in x_train]).reshape(-1, 12, 5, 1)


# Training
model = tflearn.DNN(network, tensorboard_verbose=0)
model.fit({'input': x}, {'target': y_train}, n_epoch=1, snapshot_step=100, show_metric=True, run_id='convnet_mnist')




example = "welche fachbereiche die hsd hat"
example = example.split()
sen = []

for word in example:
	sen.append(vectors[word2int[word]])
while len(sen) < 12:
	sen.append(padding)

sen = np.array(sen).reshape(1, 12, 5, 1)





classification = model.predict(sen)[0]


print(np.argmax(classification))