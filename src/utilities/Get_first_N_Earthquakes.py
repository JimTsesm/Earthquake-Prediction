#mount your Google Drive
#follow the commands prompted when executing the commands below
from google.colab import drive
drive.mount('/content/gdrive')

#Use the indexes to define which dataset you will save

!head -n 50085877 /content/data/earthquake/train.csv > /content/gdrive/My\ Drive/datasets/Eartquake_prediction/first_two_earthquakes.csv
