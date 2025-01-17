import math 
import csv
import numpy as np
import utils

def get_sub_time_serie(start, end):
	list_of_time_series = []
	time_to_failure_list = []
	acoustic_data_list = []
	with open('/content/data/earthquake/train.csv') as csvfile:
		reader = csv.DictReader(csvfile)
		previous_value = 10000.0
		for i,row in enumerate(reader):
			#if(float(row['time_to_failure'])-float(previous_value) > 5):
			#print(str(i),previous_value, row['time_to_failure'])
			#previous_value = row['time_to_failure']
			if(i >= START):
				acoustic_data_list.append(float(row['acoustic_data']))
				if(len(acoustic_data_list) == TIME_SERIES_LENGTH):
					time_to_failure_list.append(float(row['time_to_failure']))
					list_of_time_series.append(acoustic_data_list)
					acoustic_data_list = []
			if(i == END):
				break
	return np.array(time_to_failure_list), np.array(list_of_time_series)
	
def get_sub_time_serie(dataset, dateset_len, window_size, step):
  start = 1
  end = start+window_size
  print(str(start) + ' ' + str(end))
  list_of_time_series = []
  time_to_failure_list = []
  while(end <= dateset_len):
    time_to_failure_list.append(dataset[end-1][1])
    list_of_time_series.append(dataset[start:end, 0])
    start += step
    end += step
  return np.array(time_to_failure_list), np.array(list_of_time_series)


##############################################
###################   MAIN   #################

# DATASET_FILE_PATH = '/content/gdrive/My Drive/datasets/Eartquake_prediction/small.csv'
# STEPS = 1000000
# TIME_SERIES_LENGTH = 150000
# DATASET_SIZE = 20000000
# DATASET_START = 0
# DATASET_END = DATASET_SIZE

#datapoints_sum = END - START
#time_series_number = math.floor(datapoints_sum/TIME_SERIES_LENGTH)
#print(time_series_number)

#Read dataset to numpy array
#dataset = utils.read_dataset(DATASET_FILE_PATH)

#labels, sub_timeseries = get_sub_time_serie(dataset[DATASET_START:DATASET_END], DATASET_SIZE, TIME_SERIES_LENGTH, STEPS)
#print(len(labels))
#print("Labels:")
#print(str(labels[0])+" "+str(labels[1])+" "+str(labels[2]) )



	
