#Download dataset to Google Colab
!pip install -U -q kaggle
!mkdir -p ~/.kaggle
!echo '{"username":"jimtses","key":"299551a3259810064114c84d9eb438d3"}' > ~/.kaggle/kaggle.json
!chmod 600 ~/.kaggle/kaggle.json

!mkdir -p data
!kaggle competitions download -c LANL-Earthquake-Prediction -f train.csv -p data

#import libraries
import zipfile
import pandas as pd
import numpy as np

#unzip file to csv
with zipfile.ZipFile("/content/data/train.csv.zip","r") as zip_ref:
    zip_ref.extractall("/content/data/earthquake")

#read dataset with Pandas
#dt = { 'acoustic_data': 'i2', 'time_to_failure': 'f8' }
#data = pd.read_csv('/content/data/earthquake/train.csv')



#READ DATA TO CSV
import csv
with open('/content/data/earthquake/train.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    previous_value = 10000.0
    for i,row in enumerate(reader):
        if(float(row['time_to_failure'])-float(previous_value) > 5):
          print(str(i),previous_value, row['time_to_failure'])
        previous_value = row['time_to_failure']
		
