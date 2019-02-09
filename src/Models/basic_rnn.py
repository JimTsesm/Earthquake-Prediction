# TensorFlow
import tensorflow as tf
from tensorflow.contrib import rnn

# Helper libraries and utils
import sys
sys.path.append('/content/gdrive/My Drive/Earthquake_Prediction/src/utilities/')
import split_timeries
import utils

# CONSTANTS
DATASET_FILE_PATH = '/content/gdrive/My Drive/datasets/Eartquake_prediction/small.csv'
STEPS = 20000
TIME_SERIES_LENGTH = 100
DATASET_SIZE = 20000000
DATASET_START = 0
DATASET_END = DATASET_SIZE

# RNN parametres
learning_rate = 0.001
epochs = 100
inputs_num=1
output_neurons = 1
units_num = 128
batch_size = 128
layers_num = 2

xplaceholder= tf.placeholder('float',[None,TIME_SERIES_LENGTH,inputs_num])
yplaceholder = tf.placeholder('float',[None,output_neurons])

#define dnn model
def dnn_layers(input_layers):
	dense1 = tf.layers.dense(inputs=input_layers, units=64)
	return dense1

# define rnn model
def rnn_model():

	lstm_cell = tf.contrib.rnn.LSTMCell(num_units=units_num,activation=tf.nn.relu)
	cell=tf.contrib.rnn.OutputProjectionWrapper(lstm_cell,output_size=output_neurons)
	lstm_outputs,states=tf.nn.dynamic_rnn(cell,xplaceholder,dtype=tf.float32)
	
	return tf.layers.dense(tf.layers.batch_normalization(lstm_outputs),output_neurons, activation=None, kernel_initializer=tf.orthogonal_initializer())

logits = rnn_model()
print(logits)
# select loss function and optimizer
loss=tf.reduce_mean(tf.square(logits-yplaceholder))
optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

init=tf.global_variables_initializer()

#Read dataset to numpy array
#dataset = utils.read_dataset(DATASET_FILE_PATH)

#train_labels, train_inputs = split_timeries.get_sub_time_serie(dataset[DATASET_START:DATASET_END], DATASET_SIZE, TIME_SERIES_LENGTH, STEPS)
train_inputs, train_labels = utils.get_random_data()
print(len(train_labels))

train_iters = 0
if(len(train_labels) % batch_size == 0):
    train_iters = int(len(train_labels) / batch_size)
else:
    train_iters = int(len(train_labels) / batch_size) + 1

print("Optimization starting..")

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(epochs):
        epoch_loss = 0
        for i in range(train_iters):
            batch_x = utils.next_batch(train_inputs,i,batch_size)
            batch_y = utils.next_batch(train_labels,i,batch_size)       
            batch_x = batch_x.reshape((-1,TIME_SERIES_LENGTH,inputs_num))
            batch_y = batch_y.reshape((-1,output_neurons))
            _, c = sess.run([optimizer, loss], feed_dict={xplaceholder: batch_x, yplaceholder: batch_y})
            epoch_loss += c/train_iters
        print('Epoch', epoch+1, 'completed out of', epochs, 'loss:', epoch_loss)

