# TensorFlow
import tensorflow as tf
from tensorflow.contrib import rnn
#!
import numpy as np


# Helper libraries and utils
import utils

# RNN parametres
learning_rate = 0.001
epochs = 10
inputs_num=1
output_neurons = 1
units_num = 128
sequence_length = 5
batch_size = 128
layers_num = 2
#!
drop_prob = 0.4

#!
# DNN parametres
n_hidden_1 = 64

xplaceholder= tf.placeholder('float',[None,sequence_length,inputs_num])
yplaceholder = tf.placeholder('float',[None,output_neurons])

#!
def dnn(lstm_output):

	weights = {
    'h1': tf.Variable(tf.random_normal([units_num, n_hidden_1])),
    'out': tf.Variable(tf.random_normal([n_hidden_1, output_neurons]))
	}

	biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'out': tf.Variable(tf.random_normal([output_neurons]))
	}

	layer_1 = tf.add(tf.matmul(lstm_output, weights['h1']), biases['b1'])
	out_layer = tf.matmul(layer_1, weights['out']) + biases['out']

	return out_layer

#!
# define rnn model
def rnn_model():

	x = tf.unstack(xplaceholder,sequence_length,axis=1)

	lstm_cells = []
	for _ in range(layers_num):
		cell = tf.nn.rnn_cell.LSTMCell(units_num)
		cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=1-drop_prob)
		lstm_cells.append(cell)

	lstm_layers = rnn.MultiRNNCell(lstm_cells)
	outputs, states = tf.nn.static_rnn(lstm_layers, x, dtype=tf.float32)

	return dnn(outputs[-1])

logits = rnn_model()
loss=tf.reduce_mean(tf.square(logits-yplaceholder))
optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

init=tf.global_variables_initializer()

#!
train_inputs,train_labels,test_inputs,test_labels = utils.get_random_data()

train_iters = 0
if(len(train_labels) % batch_size == 0):
    train_iters = int(len(train_labels) / batch_size)
else:
    train_iters = int(len(train_labels) / batch_size + 1)

print("Optimization starting..")

with tf.Session() as sess:
	sess.run(init)

	for epoch in range(epochs):
		epoch_loss = 0

		train_inputs,train_labels = utils.suffle_data(train_inputs,train_labels)

		for i in range(train_iters):

			batch_x = utils.next_batch(train_inputs,i,batch_size)
			batch_y = utils.next_batch(train_labels,i,batch_size)

			batch_x = batch_x.reshape((-1,sequence_length,inputs_num))
			batch_y = batch_y.reshape((-1,output_neurons))

			_, c = sess.run([optimizer, loss], feed_dict={xplaceholder: batch_x, yplaceholder: batch_y})

			epoch_loss += c/train_iters

		print('Epoch', epoch+1, 'completed out of', epochs, 'loss:', epoch_loss)

	#!
	print("Testing..")

	test_batch = test_inputs.reshape((-1,sequence_length,inputs_num))
	predictions = sess.run([logits], feed_dict = {xplaceholder: test_batch})

	print('Total Mae for Test set is: ',utils.compute_error(test_labels,np.array(predictions)[0]))





