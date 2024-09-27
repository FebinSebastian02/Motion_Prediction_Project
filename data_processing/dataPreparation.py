from readDataset import dataGrabber

dataset_path = 'dataset/data/'
recording_id_sel = ['18']

# Initializing data Grabber Object
data_obj = dataGrabber(dataset_path)

data_obj.recording_id = recording_id_sel
data_obj.read_csv_with_recordingID()

track_data_raw = data_obj.get_tracks_data()
track_meta_data_raw = data_obj.get_tracksMeta_data()

print(track_data_raw)

