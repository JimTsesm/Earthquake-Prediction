import math 
import csv

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
	

NROWS = 100000000
START = 5656574
END = 50085877
#END = 6000000
TIME_SERIES_LENGTH = 150000

datapoints_sum = END - START
time_series_number = math.floor(datapoints_sum/TIME_SERIES_LENGTH)
print(time_series_number)


labels, time_series = 	get_sub_time_serie(START, END)
print(np.shape(labels))
print(np.shape(time_series))

	
