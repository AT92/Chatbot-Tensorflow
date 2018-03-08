# -*- coding: utf-8 -*-

import io
import numpy as np
import tensorflow as tf





# array with all words
words = []
#array with all sentences
sentences = []

# dictionaries with vocabulary/integers
word2int = {}
int2word = {}

# training_data for skip-gram
data = []
# The size of the window
WINDOW_SIZE = 5
# Number of iterations of training_data
n_iters = 50
# The dimension of the embedded word
EMBEDDING_DIM = 5
# The batch size for the tensorflow model
BATCH_SIZE = 512
# The learning rate of the tensorflow model
LEARNING_RATE = 0.5
# Variable for printing out the process of training_data
SKIP_STEP = 2





# function to convert numbers to one hot vectors
def to_one_hot(data_point_index, vocab_size):
    temp = np.zeros(vocab_size)
    temp[data_point_index] = 1
    return temp



with io.open('./training_data/input_data.txt', encoding="utf-8") as f:
  raw_sentences = f.readlines()


for sentence in raw_sentences:
  for word in sentence.split():
    if (not word.isdigit() and word != "|"):
        words.append(word)

words = set(words)
words = sorted(words)

vocab_size = len(words)

for i, word in enumerate(words):
  word2int[word] = i
  int2word[i] = word


for raw_sentence in raw_sentences:
    sentence = raw_sentence.replace("|", "")
    sentence_no_digits = ''.join([i for i in sentence if not i.isdigit()])
    sentences.append(sentence_no_digits.split())



for sentence in sentences:
  for word_index, word in enumerate(sentence):
    for nb_word in sentence[max(word_index - WINDOW_SIZE, 0): min(word_index + WINDOW_SIZE, len(sentence)) + 1]:
      if nb_word != word:
        data.append([word, nb_word])




x_train = []  # input word
y_train = []  # output word


print("Creating one hots")
for data_word in data:
  x_train.append(to_one_hot(word2int[data_word[0]], vocab_size))
  y_train.append(to_one_hot(word2int[data_word[1]], vocab_size))




# convert them to numpy arrays
x_train = np.asarray(x_train)
y_train = np.asarray(y_train)




x = tf.placeholder(tf.float32, shape=(None, vocab_size))
y_label = tf.placeholder(tf.float32, shape=(None, vocab_size))


W1 = tf.Variable(tf.random_normal([vocab_size, EMBEDDING_DIM]), name="w1")
b1 = tf.Variable(tf.random_normal([EMBEDDING_DIM]), name="b1") #bias
hidden_representation = tf.add(tf.matmul(x, W1), b1)

W2 = tf.Variable(tf.random_normal([EMBEDDING_DIM, vocab_size]), name="w2")
b2 = tf.Variable(tf.random_normal([vocab_size]), name="b2")
prediction = tf.nn.softmax(tf.add(tf.matmul(hidden_representation, W2), b2))

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init) #make sure you do this!
# define the loss function:
cross_entropy_loss = tf.reduce_mean(-tf.reduce_sum(y_label * tf.log(prediction), reduction_indices=[1]))
# define the training_data step:
train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy_loss)

# train for n_iter iterations
saver = tf.train.Saver()

period = round((len(x_train) / BATCH_SIZE))

print("Started")
for _ in range(n_iters):
    for i in range(period):
        batch_x = x_train[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
        batch_y = y_train[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
        sess.run(train_step, feed_dict={x: batch_x, y_label: batch_y})
        sess.run(cross_entropy_loss, feed_dict={x: batch_x, y_label: batch_y})
    if (_ % SKIP_STEP == 0):
        print("iter: ", _)



saver.save(sess, "./models/w2v.ckpt")
vectors = sess.run(W1 + b1)



import matplotlib.pyplot as plt
fig, ax = plt.subplots()
for word in words:
    print(word, vectors[word2int[word]])
    ax.annotate(word, (vectors[word2int[word]][0], vectors[word2int[word]][1] ))
plt.plot(range(6))
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()