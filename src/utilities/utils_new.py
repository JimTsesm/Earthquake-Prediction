import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import csv
import math
#!
from sklearn.metrics import mean_absolute_error



def next_batch(data, num_of_batch,batch_size):
  start_index = num_of_batch*batch_size
  end_index = num_of_batch*batch_size + batch_size
  if(end_index > len(data)):
    end_index = len(data)
  return data[start_index:end_index]

#!
def get_random_data():
	return np.random.rand(4,5),np.random.rand(4,1),np.random.rand(100,5),np.random.rand(100,1)

#!
def compute_error(labels,preds):
	assert len(labels) == len(preds)
	return mean_absolute_error(labels,preds)

#!
def suffle_data(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]