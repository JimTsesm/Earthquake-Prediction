###################################################################
#BEFORE EXECUTING THE BELOW COMMAND, READ THESE COMMENTS          #
#!!! This code requires to run download_dataset.py code !!!       #   
#After executing the commands, the file first_two_earthquakes.csv #
#will be created to datasets/Eartquake_prediction/ (Google Drive) #
###################################################################

#mount your Google Drive
#follow the commands prompted when executing the commands below
from google.colab import drive
drive.mount('/content/gdrive')

#Use the indexes to define which dataset you will save.
#Choose an index and copy-paste the number after "-n"
#Earthquake 1: 5656574 
#Earthquake 2:50085877
#Earthquake 3:104677355
#Earthquake 4:138772452
#Earthquake 5:187641819
#Earthquake 6:218652629
#Earthquake 7:245829584
#Earthquake 8:307838916
#Earthquake 9:338276286
#Earthquake 10:375377847
#Earthquake 11:419368879
#Earthquake 12:461811622
#Earthquake 13:495800224
#Earthquake 14:528777114
#Earthquake 15:585568143
#Earthquake 16:621985672
!head -n 50085877 /content/data/earthquake/train.csv > /content/gdrive/My\ Drive/datasets/Eartquake_prediction/first_two_earthquakes.csv
