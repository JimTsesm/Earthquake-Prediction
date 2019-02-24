import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

def normalize_dataset():
	dt = { 'acoustic_data': 'i2', 'time_to_failure': 'f8' }
	df = pd.read_csv('/content/data/earthquake/train.csv')
	
	acoustic_data = df[['acoustic_data']].values.astype(float)
	time_to_failure = df[['time_to_failure']].values.astype(float)
	min_max_scaler = preprocessing.MinMaxScaler()
	data_scaled = min_max_scaler.fit_transform(acoustic_data)
	
	write_dataset(data_scaled,time_to_failure,'/content/gdrive/My Drive/NormalizedDataset/','NormalizedDataset.csv')

def normalize_dataset(inputs):
    print("Start normalizing")
    min_max_scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = min_max_scaler.fit_transform(inputs)
	
    #save the scaler to file
    joblib.dump(min_max_scaler, '/content/gdrive/My Drive/Earthquake_Prediction/scalers/minmaxScaler.pkl') 

    print("End normalizing")
    return data_scaled
	
def standarize_dataset(inputs):
    print("Start standarizing")
    start = 0
    end = 1000
    standar_scaler = StandardScaler()
	
    #Partial fit to avoid memory problems
    for pass_num in range(0,math.ceil(len(inputs)/1000)):
        standar_scaler.partial_fit(inputs[start:end])
        start += 1000
        end += 1000
    data_scaled = standar_scaler.transform(inputs)
	
    #save the scaler to file
    joblib.dump(standar_scaler, '/content/gdrive/My Drive/Earthquake_Prediction/scalers/standarScaler.pkl')
    print("End standarizing")
    return data_scaled

def normalize_partial_fit(inputs, scaler, path_to_save):
    print("Partial fitting")
    scaler.partial_fit(inputs)
    #save the scaler to file
    joblib.dump(standar_scaler, path_to_save)

def standarize_partial_fit(inputs, scaler, path_to_save):
    print("Partial fitting...")
    #Partial fit to avoid memory problems
    for pass_num in range(0,math.ceil(len(inputs)/1000)):
        scaler.partial_fit(inputs[start:end])
        start += 1000
        end += 1000
    #save the scaler to file
    joblib.dump(standar_scaler, path_to_save)