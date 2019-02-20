# Helper libraries and utils
import sys
sys.path.append('/content/gdrive/My Drive/Earthquake_Prediction/src/utilities/')
sys.path.append('/content/gdrive/My Drive/Earthquake_Prediction/src/preprocessing/')
import split_timeries
import utils
import preprocess
import numpy as np

# CONSTANTS
STEPS = 5000
TIME_SERIES_LENGTH = 150000
NEW_TIME_SERIES_LENGTH = 150
DATASET_SIZE = 24429301
DATASET_START = 0
DATASET_END = DATASET_SIZE
DATASET_FILE_PATH = '/content/gdrive/My Drive/datasets/Eartquake_prediction/eq2_25m_50m.csv'
DATASET_WRITE_PATH = '/content/gdrive/My Drive/datasets/Eartquake_prediction/processed/'


#Read dataset to numpy array
dataset = utils.read_dataset(DATASET_FILE_PATH)

train_labels, train_inputs = split_timeries.get_sub_time_serie(dataset[DATASET_START:DATASET_END], DATASET_SIZE, TIME_SERIES_LENGTH, STEPS)


#Normalize dataset to 0 1
#train_inputs = preprocess.normalize_dataset(train_inputs)
train_labels = preprocess.normalize_dataset(train_labels.reshape(-1,1))
train_labels = train_labels.reshape(1,-1)[0]

#Standarize dataset and Normalize labels to 0 1
train_inputs = preprocess.standarize_dataset(train_inputs)


#Reduce time serie dimension to NEW_TIME_SERIES_LENGTH
train_inputs = utils.dim_reduction_only_mean(train_inputs, NEW_TIME_SERIES_LENGTH)
#train_inputs = utils.dim_reduction_4_features(train_inputs, NEW_TIME_SERIES_LENGTH)


#Write the new dataset
utils.write_dataset(train_inputs, train_labels, DATASET_WRITE_PATH, 'eq2_150.csv')

#Read saved dataset
train_labels, train_inputs = utils.read_dataset2(DATASET_WRITE_PATH+'eq2_150.csv')
print(len(train_labels))
print(len(train_inputs))
print(len(train_inputs[0]))