import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler


class DataPreprocessor:
    def __init__(self, track_data):
        self.ds_data = None
        self.track_meta_data_raw = None
        self.fraction = None
        self.data = track_data

    def downsample(self, fraction):
        self.fraction = fraction
        # Get every 2nd row from the dataframe
        ds_data = self.data.iloc[::(fraction + 1)]
        return ds_data

    def label_encode(self, track_meta_data):
        self.track_meta_data_raw = track_meta_data
        # creating a LabelEncoder object
        le = LabelEncoder()

        # Extracting class as an array
        cl = np.array(self.track_meta_data_raw['class'])

        # label encode the class column
        self.track_meta_data_raw['class'] = le.fit_transform(cl)

        # get the actual categorical values and their corresponding encoded values
        encoded_values = list(le.classes_)
        actual_values = list(self.track_meta_data_raw['class'].unique())
        actual_values.sort()  # To sort values in ascending order

        print("\n\nEncoded labels:- ")
        for i in range(len(encoded_values)):
            print(f'{actual_values[i]}: {encoded_values[i]}')
        self.track_meta_data_raw.to_csv('labelEncoded_data.csv', sep=',', index=False)
        return self.track_meta_data_raw

    def normalize(self, ds_data, features):
        self.ds_data = ds_data
        scaler = MinMaxScaler()
        features_to_normalize = features
        scaler.fit(self.ds_data[features_to_normalize])  # Computes min and max values
        pd.options.mode.copy_on_write = True  # To avoid copy on write warning

        # Transforming selected features
        self.ds_data[features_to_normalize] = scaler.transform(self.ds_data[features_to_normalize])
        ds_data.to_csv('normalized_data.csv', sep=',', index=False)
        return self.ds_data

    def save_to_pickle(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.ds_data, f)
            print(f'Data saved to {filename}')