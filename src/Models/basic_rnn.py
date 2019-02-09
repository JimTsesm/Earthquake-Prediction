# TensorFlow
import tensorflow as tf
from tensorflow.contrib import rnn

# Helper libraries and utils
import utils

# RNN parametres
learning_rate = 0.001
epochs = 100
inputs_num=1
output_neurons = 1
units_num = 128
sequence_length = 150
batch_size = 128
layers_num = 2

xplaceholder= tf.placeholder('float',[None,sequence_length,inputs_num])
yplaceholder = tf.placeholder('float',[None,sequence_length,output_neurons])

#define dnn model
def dnn_layers(input_layers):
	dense1 = tf.layers.dense(inputs=input_layers, units=64)
	return dense1

# define rnn model
def rnn_model():

	lstm_cell = tf.contrib.rnn.LSTMCell(num_units=units_num,activation=tf.nn.relu)
	cell=tf.contrib.rnn.OutputProjectionWrapper(lstm_cell,output_size=output_neurons)
	lstm_outputs,states=tf.nn.dynamic_rnn(cell,xplaceholder,dtype=tf.float32)
	
	return dnn_layers(lstm_outputs)

logits = rnn_model()
# select loss function and optimizer
loss=tf.reduce_mean(tf.square(logits-yplaceholder))
optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

init=tf.global_variables_initializer()

train_inputs,train_labels = utils.get_random_data()

train_iters = 0
if(len(train_labels) % batch_size == 0):
    train_iters = len(train_labels) / batch_size
else:
    train_iters = len(train_labels) / batch_size + 1

print("Optimization starting..")

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(epochs):
		epoch_loss = 0

		for i in range(train_iters):

			batch_x = utils.next_batch(train_inputs,i,batch_size)
			batch_y = utils.next_batch(train_labels,i,batch_size)

			batch_x = batch_x.reshape((-1,sequence_length,inputs_num))
			batch_y = batch_x.reshape((-1,sequence_length,output_neurons))

			_, c = sess.run([optimizer, loss], feed_dict={xplaceholder: batch_x, yplaceholder: batch_y})

			epoch_loss += c/train_iters

		print('Epoch', epoch+1, 'completed out of', epochs, 'loss:', epoch_loss)

