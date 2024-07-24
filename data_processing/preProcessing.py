"""
In this Python file, the three important data preprocessing techniques: data downsampling, label encoding, and data normalization has to be implemented. 

Sample Implementation:

class DataPreprocessor:
    def __init__(self, data):
        self.data = data
    
    def downsample(self, fraction):
        # Implement data downsampling here
        # Use 'fraction' to determine the desired downsampling ratio
        # Modify 'self.data' accordingly
        
    def label_encode(self):
        # Implement label encoding here
        # Convert categorical labels to numerical representations
        # Modify 'self.data' accordingly
        
    def normalize(self):
        # Implement data normalization here
        # Scale the numerical features to have zero mean and unit variance
        # Modify 'self.data' accordingly
        
# Example usage:
my_data = [...]  # Your dataset
preprocessor = DataPreprocessor(my_data)

preprocessor.downsample(0.5)  # Downsample the data to 50% of its original size
preprocessor.label_encode()   # Encode categorical labels
preprocessor.normalize()      # Normalize the data

PS: Recommended to return the processed data to the main.ipynb for further application
"""
import numpy as np
import pandas as pd
import pickle  # Febin1
import csv
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler


class DataPreprocessor:
    def __init__(self, track_data):
        self.ds_data = None
        self.track_meta_data_raw = None
        self.fraction = None
        #self.data = pd.DataFrame(track_data, columns=column_names_track_data)
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

        # print the actual values and their encoded values
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

        # display the minimum and maximum values for each feature
        # print('Minimum values in each feature: ', scaler.data_min_)
        # print('Maximum values in each feature:', scaler.data_max_)
        pd.options.mode.copy_on_write = True  # To avoid copy on write warning

        # Transforming selected features
        self.ds_data[features_to_normalize] = scaler.transform(self.ds_data[features_to_normalize])
        # print(self.ds_data)
        ds_data.to_csv('normalized_data.csv', sep=',', index=False)
        return self.ds_data

    def save_to_pickle(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.ds_data, f)
            print(f'Data saved to {filename}')


#global downsampled_data
#column_names_track_data = ['recordingId', 'trackId', 'frame', 'trackLifetime', 'xCenter', 'yCenter', 'heading', 'width',
                           #'length', 'xVelocity', 'yVelocity', 'xAcceleration', 'yAcceleration', 'lonVelocity',
                          # 'latVelocity', 'lonAcceleration', 'latAcceleration']
#column_names_track_meta_data = ['recordingId', 'trackId', 'initialFrame', 'finalFrame', 'numFrames', 'width', 'length',
                                #'class']
#track_data_reshaped = np.reshape(track_data_raw, (625989, 17))  # Reshape 3d to 2d object
#track_meta_data_reshaped = np.reshape(track_meta_data_raw, (427, 8))
#pd.set_option('display.float_format', '{:.6f}'.format)  # To display float value with 6 digits"""

#preprocessor = DataPreprocessor(track_data_reshaped)  # Calling data preprocessor constructor

"""downsample_data = preprocessor.downsample(1)  # Down sampling the data to 50% of its original size
preprocessor.label_encode(track_meta_data_reshaped)  # Converting categorical labels of column "class" to numerical
# values
modified_dataset = preprocessor.normalize(downsample_data)"""

# Febin2 - To be implemented after dataset is preprocessed
# To write byte data into a file
"""mds = open('modified_dataset.txt', 'wb')
pickle.dump(modified_dataset, mds)
mds.close()"""

# To read byte data
"""mds = open('modified_dataset.txt',
           'rb')  # Opens already written modified_dataset.txt and reads its content and store it in mds object.
modifiedDataSet = pickle.load(
    mds)  # The mds object is loaded to read the byte content and is stored to modifiedDataSet object
print(f"\n\nModified Dataset:- \n{modifiedDataSet}")
mds.close()  # mds object is closed
"""