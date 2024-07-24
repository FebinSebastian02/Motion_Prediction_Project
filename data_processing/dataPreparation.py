#No need to add anything here unless we are changing the dataset

#Mostly for pasting same contents from data_preparation_notebook and running it as a python file

#Importing Libraries
import numpy as np
import pandas as pd
import pickle #Febin1
from readDataset import dataGrabber

#Step 1: Reading Dataset
dataset_path = 'dataset/data/'

recording_id_sel = ['18'] #Selects recording 18

# Initializing data Grabber Object
data_obj = dataGrabber(dataset_path) #dataGrabber constructor is called and values are initialized

data_obj.recording_id = recording_id_sel #Recording id is set with value of 18
data_obj.read_csv_with_recordingID() #Reads tracks,tracks_meta and recordings csv files of recording 18

track_data_raw = data_obj.get_tracks_data()
track_meta_data_raw = data_obj.get_tracksMeta_data()

print(track_data_raw) #List with recording 18 values are obtained

