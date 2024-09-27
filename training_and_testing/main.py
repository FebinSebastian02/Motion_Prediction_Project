# This file is the main file for the lightning training of the neural network models.
# The main file is responsible for the following:
# 1. Creating the datamodule
# 2. Creating the model
# 3. Creating the callbacks
# 4. Creating the logger
# 5. Creating the trainer
# 6. Fitting the model

# READ THE COMMENTS AND TODOs CAREFULLY!
# Please check the pytorch lightning documentation for more information:
# https://lightning.ai/docs/pytorch/stable/

import logging
import lightning as pl
import torch
import wandb
import os
from callbacks import create_callbacks
from lit_datamodule import inD_RecordingModule
from utils import create_wandb_logger, get_data_path, build_module
from nn_modules import ConstantVelocityModel, MultiLayerPerceptron, ConstantAccelerationModel, LSTMModel, BicycleModel, \
    GRU
from select_features import select_features
from lit_module import LitModule

# For LSTM
import torch.backends.mkldnn as mkldnn

mkldnn.enabled = False

##################################################################

torch.set_float32_matmul_precision('medium')
torch.autograd.set_detect_anomaly(True)

# TODO: check get_data_path() in utils.py. You can change the data path there.
data_path, log_path = get_data_path()

# TODO: Check out weights and biases for logging and visualizing your results.
wandb.login()

##################################################################

project_name = "SS2024_motion_prediction"

# TODO: Choose between the stages here.
#  For Neural network models such as MLP, LSTM, GRU, the stage should be "fit" which then should be followed by "test".
#  For other models such as CVM, CAM, BCM, the stage should be "test".
#stage = "fit"
stage = "test"

#################### Training Parameters #####################################

# TODO: Choose the Recording ID you need
#  21,22 - Used for Training
#  18 - Used for Testing
recording_ID = ["18"]  # "00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14",
# "15", "16","17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32"]

# TODO: Choose the model type you want to use. The model types are defined in the nn_modules.py file.The model types are abbreviated as:-
#  a) CVM - Constant Velocity model
#  b) CAM - Constant Acceleration model
#  c) BCM - Bicycle model
#  d) MLP - Multi layer perceptron
#  e) LSTM - Long short term memory
#  f) GRU - Gated recurrent unit
model_type = "GRU"

# TODO: Change the features to the features you want to use. The features are defined in the select_features.py file
features, number_of_features, features_meta, number_of_meta_features = select_features(model_type)

if model_type == "MLP" or model_type == "LSTM" or model_type == "GRU":
    past_sequence_length = 6
    future_sequence_length = 3
else:
    past_sequence_length = 1  # for Physics based models
    future_sequence_length = 1  # for Physics based models
sequence_length = past_sequence_length + future_sequence_length

#################### Model Parameters #####################################

if model_type == "MLP":
    batch_size = 50
    input_size = number_of_features * past_sequence_length
    output_size = number_of_features
    hidden_size = 32

elif model_type == "LSTM" or model_type == "GRU":
    batch_size = 50
    input_size = number_of_features
    output_size = number_of_features
    hidden_size = 64
    num_layers = 1

else:
    batch_size = 50
    input_size = number_of_features
    output_size = number_of_features

if __name__ == '__main__':

    #################### Create Models #####################################
    # Models are created in the nn_modules.py file. The classes are inherited from the nn.Module class.
    match model_type:
        case "MLP":
            mdl = MultiLayerPerceptron(input_size, hidden_size, output_size)
        case "CVM":
            mdl = ConstantVelocityModel()
        case "CAM":
            mdl = ConstantAccelerationModel()
        case "BCM":
            mdl = BicycleModel()
        case "LSTM":
            mdl = LSTMModel(input_size, hidden_size, output_size, num_layers)
        case "GRU":
            mdl = GRU(input_size, hidden_size, output_size, num_layers)

    # In the datamodule, the dataset is created. The dataset is created using the inD_RecordingDataset class.
    # The data set is defined in the lit_dataset.py file.

    dm = inD_RecordingModule(data_path, recording_ID, sequence_length, past_sequence_length, future_sequence_length,
                             features, features_meta, stage, model_type, batch_size=batch_size)

    #################### Setup Training #####################################

    # TODO: Change the epochs to the number of epochs you want to train
    epochs = 1

    # In the lit_module.py file, the model is defined. The model is defined using the LitModule class.
    model = LitModule(mdl, number_of_features, sequence_length, past_sequence_length, future_sequence_length,
                      batch_size)
    dm.setup(stage=stage)

    # TODO: Change the callbacks to the callbacks you want to use. The callbacks are defined in the callbacks.py file
    callbacks = create_callbacks(model_type)
    wandb_logger = create_wandb_logger(log_path, project_name, recording_ID)
    wandb_logger.experiment.config.update({"batch_size": batch_size,
                                           "sequence_length": sequence_length})
    logging.getLogger(log_path + "/lightning").setLevel(logging.ERROR)

    #################### Start Training #####################################

    trainer = pl.Trainer(max_epochs=epochs,
                         fast_dev_run=False,
                         devices="auto",
                         accelerator="auto",
                         log_every_n_steps=5,
                         logger=wandb_logger,
                         callbacks=callbacks,
                         check_val_every_n_epoch=1,
                         precision="16-mixed"
                         # precision="32"
                         )

    if stage == "fit":
        trainer.fit(model, dm)
    elif stage == "test":
        if model_type == "MLP" or model_type == "LSTM" or model_type == "GRU":
            model_params = {
                'model': mdl,
                'number_of_features': number_of_features,
                'sequence_length': sequence_length,
                'past_sequence_length': past_sequence_length,
                'future_sequence_length': future_sequence_length,
                'batch_size': batch_size
            }
            if model_type == "GRU":
                ckpt_file_path = os.path.join(log_path, 'GRU-checkpoint.ckpt')
            elif model_type == "MLP":
                ckpt_file_path = os.path.join(log_path, 'MLP-checkpoint.ckpt')
            else:
                ckpt_file_path = os.path.join(log_path, 'LSTM-checkpoint.ckpt')
            if os.path.exists(ckpt_file_path):
                checkpoint = torch.load(ckpt_file_path)
            else:
                raise FileNotFoundError(f"Checkpoint not found at {ckpt_file_path}")

            print(f"Checkpoint keys :- {checkpoint['state_dict'].keys()}")  # Checkpoint keys
            print(f"Model keys :- {model.state_dict().keys()}")  # Model keys
            model1 = LitModule.load_from_checkpoint(ckpt_file_path, **model_params, strict=False)
            trainer.test(model1, dm)
            mdl.plot_results()
        else:
            trainer.test(model, dm)
            # TODO: Check test_loss_results folder to see the csv files of the predicted values and the actual values
            mdl.plot_results()

    wandb.finish()
