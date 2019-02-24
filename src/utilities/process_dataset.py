# Helper libraries and utils
import sys
sys.path.append('/content/gdrive/My Drive/Earthquake_Prediction/src/utilities/')
sys.path.append('/content/gdrive/My Drive/Earthquake_Prediction/src/preprocessing/')
import split_timeries
import utils
import preprocess
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

# CONSTANTS
STEPS = 5000
TIME_SERIES_LENGTH = 150000
NEW_TIME_SERIES_LENGTH = 150
DATASET_SIZE = 24429301
DATASET_START = 0
DATASET_END = DATASET_SIZE
DATASET_FILE_PATH1 = '/content/gdrive/My Drive/datasets/Eartquake_prediction/eq2_first_half.csv'
DATASET_WRITE_PATH1 = '/content/gdrive/My Drive/datasets/Eartquake_prediction/processed/output1.csv'
DATASET_FILE_PATH2 = '/content/gdrive/My Drive/datasets/Eartquake_prediction/eq2_second_half.csv'
DATASET_WRITE_PATH2 = '/content/gdrive/My Drive/datasets/Eartquake_prediction/processed/output2.csv'

#OPERATION PARAMS

#TYPE=1 FOR PARTIAL FIT TYPE=2 FOR TRANSFORM
OPERATION_TYPE = 1
EARTH_QUAKE_NUM = 2
EARTH_QUAKE_PART = 1
SCALER1_PATH = '/content/gdrive/My Drive/Earthquake_Prediction/scalers/minmaxScaler_'+str(EARTH_QUAKE_NUM)+'.pkl'
SCALER2_PATH = '/content/gdrive/My Drive/Earthquake_Prediction/scalers/standarscalerScaler_'+str(EARTH_QUAKE_NUM)+'.pkl'

####################################################################
#One Earthquake can't be loaded to RAM so we need to load 
#and process each half of the Earthquake independently. 
#We need to partially Normalize/Standarize and then reload each file 
#and Transform and produce an output
#containing the reduced dataset with the merged inputs.
####################################################################

if OPERATION_TYPE == 1:
	if EARTH_QUAKE_PART == 1:
		#Partial fit with the first half of the dataset
		dataset = utils.read_dataset(DATASET_FILE_PATH1)

		min_max_scaler = MinMaxScaler(feature_range=(0, 1))
		min_max_scaler = preprocess.normalize_partial_fit(np.array(dataset[:,1]).reshape(-1,1), min_max_scaler, SCALER1_PATH)
		standar_scaler = StandardScaler()
		standar_scaler = preprocess.standarize_partial_fit(np.array(dataset[:,0]).reshape(-1,1), standar_scaler, SCALER2_PATH)

	else:
		min_max_scaler = joblib.load(SCALER1_PATH)
		standar_scaler = joblib.load(SCALER2_PATH)
		#Partial fit with the first half of the dataset
		if(DATASET_FILE_PATH2 != ''):
			dataset = utils.read_dataset(DATASET_FILE_PATH2)
			min_max_scaler = preprocess.normalize_partial_fit(np.array(dataset[:,1]).reshape(-1,1), min_max_scaler,SCALER1_PATH)
			standar_scaler = preprocess.standarize_partial_fit(np.array(dataset[:,0]).reshape(-1,1), standar_scaler, SCALER2_PATH)

#if OPERATION_TYPE == 2:
##################################
#Process first half of the dataset
##################################

	#Read saved scalers
	#min_max_scaler = joblib.load(SCALER1_PATH)
	#standar_scaler = joblib.load(SCALER2_PATH)

	# #Read dataset to numpy array
	# dataset = utils.read_dataset(DATASET_FILE_PATH1)

	# #Generate new dataset with sub timeseries
	# train_labels, train_inputs = split_timeries.get_sub_time_serie(dataset[DATASET_START:DATASET_END], DATASET_SIZE, TIME_SERIES_LENGTH, STEPS)

	# #Normalize dataset to 0 1
	# #train_inputs = preprocess.normalize_dataset(train_inputs)
	# train_labels = preprocess.normalize_dataset(train_labels.reshape(-1,1))
	# train_labels = train_labels.reshape(1,-1)[0]

	# #Standarize dataset and Normalize labels to 0 1
	# train_inputs = preprocess.standarize_dataset(train_inputs)


	# #Reduce time serie dimension to NEW_TIME_SERIES_LENGTH
	# train_inputs = utils.dim_reduction_only_mean(train_inputs, NEW_TIME_SERIES_LENGTH)
	# #train_inputs = utils.dim_reduction_4_features(train_inputs, NEW_TIME_SERIES_LENGTH)


	# #Write the new dataset
	# utils.write_dataset(train_inputs, train_labels, DATASET_WRITE_PATH1, 'eq2_150.csv')