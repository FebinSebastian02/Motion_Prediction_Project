import lightning as pl
import torch
from torch.utils.data import DataLoader, random_split
from lit_dataset import *

class inD_RecordingModule(pl.LightningDataModule):
    """LightningDataModule for inD dataset.
    Parameters
    ----------
    data_path : str
        Path to the data.
    recording_id : int
        Recording id of the data.
    sequence_length : int
        Length of the sequence.
    past_sequence_length : int
        Length of the past sequence.
    future_sequence_length : int
        Length of the future sequence.
    features : list
        List of features to use.
    batch_size : int
        Batch size.
    """

    def __init__(self, data_path, recording_id,
                 sequence_length, past_sequence_length, future_sequence_length, features, features_meta, stage, model_type,
                 batch_size: int = 32):
        super().__init__()
        self.data_path = data_path
        self.recording_id = recording_id
        self.batch_size = batch_size
        self.transform = None
        self.sequence_length = sequence_length
        self.past_sequence_length = past_sequence_length
        self.future_sequence_length = future_sequence_length
        self.features = features
        self.features_meta = features_meta
        self.stage = stage
        self.model_type = model_type
        self.save_hyperparameters()

    def setup(self, stage: str):
        """Setup the data.
        Parameters
        ----------
        stage : str
            Stage of the data. Can be "fit", "test" or "predict".
        """
        if stage == "test":
            # self.test = inD_RecordingDataset(self.data_path, self.recording_id, self.sequence_length,
            # self.features, self.transform)
            self.test = inD_RecordingDataset(self.data_path, self.recording_id, self.sequence_length, self.features,
                                             self.features_meta, self.stage, self.model_type, self.transform)
            data_size = len(self.test)
            print(f"Data size:- {data_size}")

        if stage == "predict":
            self.predict = inD_RecordingDataset(self.data_path, self.recording_id, self.sequence_length, self.features,
                                                self.transform)
        if stage == "fit":
            full = inD_RecordingDataset(self.data_path, self.recording_id, self.sequence_length, self.features,
                                        self.features_meta,  self.stage, self.model_type, self.transform)
            data_size = len(full)
            print(f"Data size:- {data_size}")
            # TODO: change the ration between train and val if you like!
            train_size = int(0.95 * data_size)
            val_size = int(data_size - train_size)
            print(f"Training size:- {train_size}, Validation size:- {val_size}")
            self.train, self.val = random_split(full, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(self.train,
                          batch_size=self.batch_size,
                          num_workers=os.cpu_count() - 2,
                          persistent_workers=True,
                          shuffle=True,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val,
                          batch_size=self.batch_size,
                          num_workers=os.cpu_count() - 2,
                          persistent_workers=True,
                          shuffle=False,
                          pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test,
                          batch_size=self.batch_size,
                          num_workers=os.cpu_count() - 2,
                          pin_memory=True)

    def predict_dataloader(self):
        return DataLoader(self.predict,
                          batch_size=self.batch_size,
                          num_workers=os.cpu_count() - 2,
                          pin_memory=True)

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        print("tear down")
