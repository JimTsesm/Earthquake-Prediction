import numpy as np
import pandas as pd

def normalize_dataset():
	dt = { 'acoustic_data': 'i2', 'time_to_failure': 'f8' }
	df = pd.read_csv('/content/data/earthquake/train.csv')
	
	acoustic_data = df[['acoustic_data']].values.astype(float)
	time_to_failure = df[['time_to_failure']].values.astype(float)
	min_max_scaler = preprocessing.MinMaxScaler()
	data_scaled = min_max_scaler.fit_transform(acoustic_data)
	
	write_dataset(data_scaled,time_to_failure,'/content/gdrive/My Drive/NormalizedDataset/','NormalizedDataset.csv')

def normalize_dataset(train_inputs):
	data_scaled = min_max_scaler.fit_transform(train_inputs)
	return data_scaled
