# Helper libraries and utils
import sys
sys.path.append('/content/gdrive/My Drive/Earthquake_Prediction/src/utilities/')
import split_timeries
import utils
import numpy as np

# CONSTANTS
STEPS = 200000
TIME_SERIES_LENGTH = 150000
DATASET_SIZE = 20000000
DATASET_START = 0
DATASET_END = DATASET_SIZE
DATASET_FILE_PATH = '/content/gdrive/My Drive/datasets/Eartquake_prediction/small.csv'
DATASET_WRITE_PATH = '/content/gdrive/My Drive/datasets/Eartquake_prediction/'


#Read dataset to numpy array
#dataset = utils.read_dataset(DATASET_FILE_PATH)

#train_labels, train_inputs = split_timeries.get_sub_time_serie(dataset[DATASET_START:DATASET_END], DATASET_SIZE, TIME_SERIES_LENGTH, STEPS)

#Write the new dataset
#utils.write_dataset(train_inputs, train_labels, DATASET_WRITE_PATH, 'splited_150000.csv')

#Read saved dataset
train_labels, train_inputs = utils.read_dataset2(DATASET_WRITE_PATH+'splited_150000.csv')
print(len(train_labels))
print(len(train_inputs))
print(len(train_inputs[0]))