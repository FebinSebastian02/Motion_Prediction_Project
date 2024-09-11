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
        #self.trajectories = {"ground_truth": [], "predicted": []}  # Febin
        # A dictionary was created to store values of ground truth and predicted values in list with keys being
        # ground_truth and predicted

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
        x = x.float() # Ensure x is in FP32 #Febin
        y = y.float()  # Ensure y is in FP32 #Febin


        y_hat_list = []
        for k in range(self.future_sequence_length):
            y_hat_k = self(x)
            y_hat_list.append(y_hat_k)
            if y_hat_k.dim() < 3:
                y_hat_k = y_hat_k.unsqueeze(1)
            # x = torch.cat([x[:, 1:, :], y_hat_k], dim=1)
            x = torch.cat([x[:, 1:, :].float(), y_hat_k.float()], dim=1) #Febin
            #print(f"x dtype: {x.dtype}, y_hat_k dtype: {y_hat_k.dtype}") #Febin

        y_hat = torch.stack(y_hat_list, dim=1).squeeze()

        loss = self.model.loss_function(y_hat, y)

        self.log(f"{string}_loss", loss)

        #trajectories = [y, y_hat] #Febin
        #self.plotGraph(trajectories) #Febin

        return loss
        #self.trajectories = [y, y_hat]  # Febin
        # trajectories = model.get_trajectories()
        #self.plotGraph(self.trajectories)  # Febin

        #Febin

        # Ensure y and y_hat have at least 2 dimensions for proper concatenation later
        """   if y.dim() == 1:
            y = y.unsqueeze(1)
        if y_hat.dim() == 1:
            y_hat = y_hat.unsqueeze(1)

        # Append to trajectories
        self.trajectories['ground_truth'].append(y.cpu().numpy())
        self.trajectories['predicted'].append(y_hat.cpu().numpy())"""

    def prep_data_for_step(self, batch):
        # TODO: This is a hacky way to load one rectangular block from the data, and divide it into x and y of different
        #  sizes afterwards.
        #  If you don't do it like this, you run into trouble. Just stay aware of this.
        #x = batch[:, :self.sequence_length, :] #febin
        #x = batch[:, :self.past_sequence_length, :]
        #y = batch[:, self.sequence_length:, :] #febin
        #y = batch[:, self.past_sequence_length:, :]

        x = batch[:, :self.past_sequence_length, :]  # Use past_sequence_length for x in lstm also
        #y = batch[:, self.past_sequence_length:self.past_sequence_length + self.future_sequence_length,
         #   :]  # Use future_sequence_length for y for lstm only
        y = batch[:, self.past_sequence_length:, :]
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


    """ # Febin"""
"""  def get_trajectories(self):
        return self.trajectories"""

    # Febin

    # New method to aggregate trajectories after the test phase
"""    def aggregate_trajectories(self):
        ground_truth = np.concatenate(self.trajectories['ground_truth'], axis=0)
        predicted = np.concatenate(self.trajectories['predicted'], axis=0)

        # Ensure ground_truth and predicted are 2D arrays with at least 2 columns
        if ground_truth.ndim == 3 and ground_truth.shape[1] == 1:
            ground_truth = ground_truth.squeeze(axis=1)

        # Reshape if necessary to ensure at least 2 columns
        if ground_truth.ndim == 1:
            ground_truth = ground_truth.reshape(-1, 2)
        if predicted.ndim == 1:
            predicted = predicted.reshape(-1, 2)

        if ground_truth.shape[1] < 2 or predicted.shape[1] < 2:
            raise ValueError("Trajectories must have at least 2 columns for x and y coordinates.")

        return ground_truth, predicted

    def plot_trajectories(self, ground_truth, predicted, num_samples=100):
        # Ensure ground_truth and predicted are numpy arrays
        if isinstance(ground_truth, torch.Tensor):
            ground_truth = ground_truth.cpu().numpy()
        if isinstance(predicted, torch.Tensor):
            predicted = predicted.cpu().numpy()

        # Ensure ground_truth and predicted have at least 2 columns
        assert ground_truth.shape[1] >= 2 and predicted.shape[
            1] >= 2, "Trajectories must have at least 2 columns for x and y coordinates."

        # Select a subset of the trajectories
        if ground_truth.shape[0] > num_samples:
            indices = np.random.choice(ground_truth.shape[0], num_samples, replace=False)
        else:
            indices = np.arange(ground_truth.shape[0])

        ground_truth_subset = ground_truth[indices]
        predicted_subset = predicted[indices]

        # Plot the subset of trajectories
        fig, ax = plt.subplots(figsize=(10, 8))
        for gt, pred in zip(ground_truth_subset, predicted_subset):
            ax.plot(gt[:, 0], gt[:, 1], color='b', alpha=0.5)
            ax.plot(pred[:, 0], pred[:, 1], color='r', alpha=0.5)

        # Plot the mean trajectory
        mean_ground_truth = np.mean(ground_truth, axis=0)
        mean_predicted = np.mean(predicted, axis=0)
        ax.plot(mean_ground_truth[:, 0], mean_ground_truth[:, 1], color='b', linewidth=2, label='Mean Ground Truth')
        ax.plot(mean_predicted[:, 0], mean_predicted[:, 1], color='r', linewidth=2, label='Mean Predicted')

        ax.set_xlabel('X position (m)')
        ax.set_ylabel('Y position (m)')
        ax.set_title('Bicycle Trajectory')
        ax.legend()
        ax.grid(True)
        plt.show()"""


