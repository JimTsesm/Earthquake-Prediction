import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import csv
import math
from numpy import genfromtxt


def read_dataset(path):
	print("Start reading "+path)
	dataset = genfromtxt(path, delimiter = ',')
	print("End reading "+path+"\n\n")
	return dataset


def next_batch(data, num_of_batch,batch_size):
  start_index = num_of_batch*batch_size
  end_index = num_of_batch*batch_size + batch_size
  if(end_index > len(data)):
    end_index = len(data)
  return data[start_index:end_index]


def get_random_data():
	return np.random.rand(1000,100),np.random.rand(1000,1)