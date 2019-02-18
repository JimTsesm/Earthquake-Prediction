import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import csv
import math
import datetime
from numpy import genfromtxt
from sklearn.metrics import mean_absolute_error


def read_dataset(path):
	print("Start reading "+path)
	dataset = genfromtxt(path, delimiter = ',')
	print("End reading "+path+". Read "+str(len(dataset))+" lines."+"\n\n")
	return dataset

def read_dataset2(path):
	print("Start reading "+path)
	dataset = genfromtxt(path, delimiter = ',')
	print("End reading "+path+"\n\n")
	train_inputs = []
	train_labels = []
	for i in range(0,len(dataset)):
		train_inputs.append(np.array(dataset[i][0:len(dataset[i])-1]))
		train_labels.append(dataset[i][-1])
	print("End reading "+path+". Read "+str(len(train_labels))+" lines."+"\n\n")
	return np.array(train_labels), np.array(train_inputs)

def write_dataset(train_inputs, train_labels, path, file_name):
	w = csv.writer(open(path+file_name, "w"))
	for i in range(0,len(train_labels)):
		l = []
		for j in range(0,len(train_inputs[i])):
			l.append(str(train_inputs[i][j]))
		l.append(str(train_labels[i]))
		w.writerow(l)
	

def next_batch(data, num_of_batch,batch_size):
  start_index = num_of_batch*batch_size
  end_index = num_of_batch*batch_size + batch_size
  if(end_index > len(data)):
    end_index = len(data)
  return data[start_index:end_index]
  

def dim_reduction(data,new_len):
	print("Start length reduction")
	window = int(len(data[0]) / new_len)	
	result = []
	for timeserie in data:
		reduced_timeserie = []
		for index in range(0,new_len):
			start_index = index*window
			end_index = index*window + window

			reduced_timeserie.append(np.mean(timeserie[start_index:end_index]))

		result.append(np.array(reduced_timeserie))
	print("End length reduction")
	return np.array(result)


def get_random_data():
	return np.random.rand(100,1500),np.random.rand(100,1)

#def get_random_data():
#	return np.random.rand(100,1500),np.random.rand(100,1),np.random.rand(100,1500),np.random.rand(100,1)

def compute_error(labels,preds):
	assert len(labels) == len(preds)
	return mean_absolute_error(labels,preds)

def suffle_data(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def plot_epoch_loss(epoch_loss_list, epoch_index_list, path):
	plt.ylabel('Epoch loss')
	plt.xlabel('Epoch #')
	plt.plot(epoch_index_list, epoch_loss_list)
	plt.savefig(path+'plot_'+str(datetime.datetime.now())+'.jpeg')

