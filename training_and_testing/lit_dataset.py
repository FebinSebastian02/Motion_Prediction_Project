import torch
import os
from torch.utils.data import Dataset
from torchvision import transforms
from data_processing.preProcessing import *


class inD_RecordingDataset(Dataset):
    def __init__(self, path, recording_id, sequence_length, features, features_meta, stage, model_type, train=True):
        """Dataset for inD dataset.
        Parameters
        ----------
        path : stri
            Path to the data.
        recording_id : int
            Recording id of the data.
        sequence_length : int
            Length of the sequence.
        features : list
            List of features to use.
        train : bool
            Whether to use the training set or not.
        """
        super(inD_RecordingDataset).__init__()
        self.path = path
        self.recording_id = recording_id
        self.sequence_length = sequence_length
        self.features = features
        self.features_meta = features_meta
        self.stage = stage
        self.model_type = model_type
        self.train = train
        self.transform = self.get_transform()

        pickle_filename = f"{path}/{recording_id}_processed_data.pkl"

        if model_type == "MLP" or model_type == "LSTM" or model_type == "GRU":
            # Load processed data if it exists
            if os.path.exists(pickle_filename):
                print("Loading processed data from pickle file.")
                with open(pickle_filename, 'rb') as f:
                    self.data = pickle.load(f)
                    print(self.data)

            else:
                print("Processing data...")

                if type(self.recording_id) == list:
                    self.data = pd.DataFrame()
                    self.meta_data = pd.DataFrame()  # Creating an empty panda dataframe for meta files
                    # TODO: Here we are simply loading the csv and stack them into one pandas dataframe.
                    for id in self.recording_id:
                        with open(f"{path}/{id}_tracks.csv", 'rb') as f:
                            self.data = pd.concat([self.data,
                                                   pd.read_csv(f, delimiter=',', header=0, usecols=self.features,
                                                               dtype='float16')])
                    # For meta files
                    for id in self.recording_id:
                        with open(f"{path}/{id}_tracksMeta.csv", 'rb') as f:
                            self.meta_data = pd.concat(
                                [self.meta_data, pd.read_csv(f, delimiter=',', header=0, usecols=self.features_meta)])
                    # For data processing
                    print(f"\n///Raw tracks_data/// \n {self.data}")
                    print(f"\n///Raw tracks_meta_data/// \n {self.meta_data}")
                    preprocessor = DataPreprocessor(self.data)  # Calling data preprocessor constructor
                    downsample_data = preprocessor.downsample(1)  # Down sampling the data to 50% of its original size
                    print(f"\n///Downsampled_data/// \n {downsample_data}")
                    labelEncoded_data = preprocessor.label_encode(
                        self.meta_data)  # Converting categorical labels of column "class" to numerical
                    # values
                    print(f"\n///Labelencoded_data/// \n {labelEncoded_data}")
                    normalized_data = preprocessor.normalize(downsample_data, self.features)
                    print(f"\n///Normalized_data/// \n {normalized_data}")
                    print("\nData written to CSV file successfully.")
                    preprocessor.save_to_pickle(pickle_filename)
                    self.data = normalized_data

                else:
                    with open(f"{path}/{recording_id}_tracks.csv", 'rb') as f:
                        self.data = pd.read_csv(f, delimiter=',', header=0, usecols=self.features, dtype='float16')

        elif model_type == "CVM" or model_type == "CAM" or model_type == "BCM":

            if type(self.recording_id) == list:
                self.data = pd.DataFrame()
                self.meta_data = pd.DataFrame()  # Creating an empty panda dataframe for meta files
                # TODO: Here we are simply loading the csv and stack them into one pandas dataframe.
                for id in self.recording_id:
                    with open(f"{path}/{id}_tracks.csv", 'rb') as f:
                        self.data = pd.concat([self.data,
                                               pd.read_csv(f, delimiter=',', header=0, usecols=self.features,
                                                           dtype='float16')])
                # For meta files
                for id in self.recording_id:
                    with open(f"{path}/{id}_tracksMeta.csv", 'rb') as f:
                        self.meta_data = pd.concat(
                            [self.meta_data, pd.read_csv(f, delimiter=',', header=0, usecols=self.features_meta)])

                print(f"\n///Raw tracks_data/// \n {self.data}")
                print(f"\n///Raw tracks_meta_data/// \n {self.meta_data}")

            else:
                with open(f"{path}/{recording_id}_tracks.csv", 'rb') as f:
                    self.data = pd.read_csv(f, delimiter=',', header=0, usecols=self.features, dtype='float16')

    def __len__(self):
        """
        Returns the length of the dataset.
        """
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        """
                 Returns the item at index idx.
        Parameters
        ----------
        idx : int
            Index of the item.
        Returns
        -------
        data : torch.Tensor
            The data at index idx.
        """
        if idx <= self.__len__():
            data = self.data[idx:idx + self.sequence_length]

            if self.transform:
                data = self.transform(np.array(data, dtype='float16')).squeeze()
            return data
        else:
            print("wrong index")
            return None

    def get_transform(self):
        """
        Returns the transform for the data.
        """
        data_transforms = transforms.Compose([
            transforms.ToTensor(),
        ])
        return data_transforms
