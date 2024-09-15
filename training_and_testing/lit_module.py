import lightning as pl
import torch
import torch.nn as nn
import csv  # Febin
import torch.nn.functional as F
from matplotlib import pyplot as plt  # Febin
import numpy as np
from nn_modules import ConstantVelocityModel
from nn_modules import MultiLayerPerceptron


class LitModule(pl.LightningModule):
    """
    This is a standard PyTorch Lightning module,
    with a few PyTorch-Lightning-specific things added.

    The main things to notice are:
    - Instead of defining different steps for training, validation, and testing,
        we define a single `step` function, and then define `training_step`,
        `validation_step`, and `test_step` as thin wrappers that call `step`.
    """

    def __init__(self, model, number_of_features, sequence_length, past_sequence_length, future_sequence_length,
                 batch_size):
        super().__init__()
        self.model = model
        self.nx = number_of_features
        self.sequence_length = sequence_length
        self.past_sequence_length = past_sequence_length
        self.future_sequence_length = future_sequence_length
        self.batch_size = batch_size

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        string = "training"
        loss = self.step(batch, batch_idx, string)
        return loss

    def validation_step(self, batch, batch_idx):
        string = "validation"
        loss = self.step(batch, batch_idx, string)
        return loss

    def test_step(self, batch, batch_idx):
        string = "test"
        loss = self.step(batch, batch_idx, string)
        return loss

    def step(self, batch, batch_idx, string):
        """
        This is the main step function that is used by training_step, validation_step, and test_step.
        """
        # TODO: You have to modify this based on your task, model and data. This is where most of the engineering
        #  happens!
        x, y = self.prep_data_for_step(batch)
        x = x.float()  # Ensure x is in FP32 #Febin
        y = y.float()  # Ensure y is in FP32 #Febin

        y_hat_list = []

        for k in range(self.future_sequence_length):
            y_hat_k = self(x)
            y_hat_list.append(y_hat_k)
            if y_hat_k.dim() < 3:
                y_hat_k = y_hat_k.unsqueeze(1)
            x = torch.cat([x[:, 1:, :].float(), y_hat_k.float()], dim=1)  # Febin

        y_hat = torch.stack(y_hat_list, dim=1).squeeze()

        loss = self.model.loss_function(y_hat, y)

        self.log(f"{string}_loss", loss)


        #For plot
        # Store predictions and ground truth for plotting
        if batch_idx == 0:

            self.y_pred = y_hat
            self.y_true = y
        else:


            self.y_pred = torch.cat((self.y_pred, y_hat), dim=0)
            self.y_true = torch.cat((self.y_true, y), dim=0)
        return loss

    def plot_predictions(self, y_pred, y_true, downsample_factor=10000):
        # Ensure y_pred and y_true are detached, moved to CPU, and cast to float32
        y_pred = y_pred.detach().cpu().to(torch.float32).numpy()
        y_true = y_true.detach().cpu().to(torch.float32).numpy()

        # Debug statements to check dimensions
        print(f"y_pred shape: {y_pred.shape}")
        print(f"y_true shape: {y_true.shape}")

        # Reshape y_pred and y_true to 2-dimensional arrays
        if y_pred.ndim == 3 or y_true.ndim == 3:
            y_pred = y_pred.reshape(-1, y_pred.shape[-2] * y_pred.shape[-1])
            y_true = y_true.reshape(-1, y_true.shape[-2] * y_true.shape[-1])
        # Reshape y_true if it has an extra dimension
        if y_true.ndim == 3 and y_true.shape[1] == 1:
            y_true = y_true.squeeze(1)

        # Downsample to plot fewer points if the dataset is too large
        if y_pred.shape[0] > downsample_factor:
            indices = np.arange(0, y_pred.shape[0], downsample_factor)
            y_pred = y_pred[indices]
            y_true = y_true[indices]
        if y_pred.ndim != 2 or y_true.ndim != 2:
            raise ValueError("Expected y_pred and y_true to be 2-dimensional arrays")
        # Check if y_pred and y_true are 2-dimensional
        if y_pred.ndim == 2 and y_true.ndim == 2:
            # Plot x predictions vs x ground truth
            plt.figure(figsize=(14, 7))

            plt.subplot(1, 2, 1)
            plt.plot(y_true[:, 0].flatten(), label='x_true', color='blue', linestyle='-', alpha=0.6, linewidth=0.5)
            plt.plot(y_pred[:, 0].flatten(), label='x_pred', color='red', linestyle='--', alpha=0.6, linewidth=0.5)
            plt.xlabel('Time Step')
            plt.ylabel('X Position')
            plt.legend()
            plt.title('X Position: Predicted vs Ground Truth')
            plt.grid(True)

            # Check if there is a second column to plot for y
            if y_true.shape[1] > 1 and y_pred.shape[1] > 1:
                # Plot y predictions vs y ground truth
                plt.subplot(1, 2, 2)
                plt.plot(y_true[:, 1].flatten(), label='y_true', color='blue', linestyle='-', alpha=0.6, linewidth=1)
                plt.plot(y_pred[:, 1].flatten(), label='y_pred', color='red', linestyle='--', alpha=0.6, linewidth=1)
                plt.xlabel('Time Step')
                plt.ylabel('Y Position')
                plt.legend()
                plt.title('Y Position: Predicted vs Ground Truth')
                plt.grid(True)

            plt.tight_layout()
            plt.show()
        else:
            raise ValueError("Expected y_pred and y_true to be 2-dimensional arrays")

    def prep_data_for_step(self, batch):
        # TODO: This is a hacky way to load one rectangular block from the data, and divide it into x and y of different
        #  sizes afterwards.
        #  If you don't do it like this, you run into trouble. Just stay aware of this.
        x = batch[:, :self.past_sequence_length, :]
        y = batch[:, self.past_sequence_length:, :]
        #y = batch[:, self.past_sequence_length:self.past_sequence_length + self.future_sequence_length,
         #   :]  # Use future_sequence_length for y for lstm only
        return x, y

    def configure_optimizers(self):
        parameters = [p for p in self.parameters() if p.requires_grad]
        # if parameters: #Previus code
        # if parameters: #Febin
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=1e-3,
                                     weight_decay=1e-3,
                                     eps=1e-5,
                                     # fused=True,
                                     fused=False,  # Febin
                                     amsgrad=True)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=0.2,
            patience=3,
            threshold=1e-4,
            cooldown=2,
            eps=1e-6,
            verbose=True,
        )
        optimizer_and_scheduler = {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "training_loss",
                "frequency": 1,
                "strict": True}
        }
        return optimizer_and_scheduler

        # Previous code
        # else:
        #     return []
        # else: #Febin
        # return [] #Febin
            # return None #Febin - modified for CVM error


