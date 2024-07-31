import time

import lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from utils import build_module

# TODO: Here you should create your models. You can use the MLPModel or ConstantVelocity as a template.
#  Each model should have a __init__ function, a forward function, and a loss_function function.
#  The loss function doen't have to be in the model, but it is convenient to have it there, because the lit_module
#  will call it automatically, because you assign a prediction model to it and later it asks the model for the loss function.
import torch.nn as nn


class ConstantVelocityModel(nn.Module):
    def __init__(self, dt=1.0):
        super(ConstantVelocityModel, self).__init__()
        self.dt = dt

    def forward(self, x):
        #print(f"Value of x1:- {x}")
        #time.sleep(10)
        x = x[:, -1, :] ## Take the last element in the sequence
        #print(f"Value of x2:- {x}")
        #time.sleep(1)


        #Febin
        # The input tensor structure is [x_position, y_position, x_velocity, y_velocity]
        x_position = x[:, :2]  # First two columns for positions
        x_velocity = x[:, 2:]  # Last two columns for velocities

       # print(f"Value of x_position:- {x_position}")
        #time.sleep(10)

       # print(f"Value of x_velocity:- {x_velocity}")
        #time.sleep(1)

        #x_plus = x + self.dt * x

        #Febin
        # Update positions based on velocities and time step
        x_plus_position = x_position + self.dt * x_velocity
        #print(f"Value of updated position:- {x_plus_position}")
        #time.sleep(1)

        # Combine updated positions with original velocities
        x_plus = torch.cat((x_plus_position, x_velocity), dim=1) #velocity is constant and positions gets updated. Needed to avoid tensor size mismatch error too
        #print(f"Value of concatenated variable:- {x_plus}")
        #time.sleep(1)
        return x_plus

    #Febin
    def loss_function(self, predictions, targets):
        #print("inside constant velocity loss fn")

        # Ensure predictions and targets have the same shape
        #targets = targets.view_as(predictions) #Febin
        targets = targets.squeeze(1) #Febin - To change shape from [43, 1, 12] to [43,12]
        """print(f"Shape of targets:- {targets.shape}")
        print(f"Predicted value: {predictions}")
        print(f"Target value: {targets}")"""
        return F.mse_loss(predictions, targets)


class ConstantAccelerationModel(nn.Module):
    def __init__(self, dt = 1.0):
        super(ConstantAccelerationModel, self).__init__()
        self.dt = dt  # Time step

    def forward(self, x):
        x = x[:, -1, :]  # Take the last element in the sequence

        # Split into position, velocity, and acceleration components
        x_position = x[:, :2]  # First two columns for positions
        x_velocity = x[:, 2:4]  # Next two columns for velocities
        x_acceleration = x[:, 4:]  # Last two columns for accelerations

        # Update positions based on velocities, accelerations, and time step
        x_plus_position = x_position + self.dt * x_velocity + 0.5 * self.dt ** 2 * x_acceleration
        # Update velocities based on accelerations and time step
        x_plus_velocity = x_velocity + self.dt * x_acceleration

        # Combine updated positions, updated velocities, and original accelerations
        x_plus = torch.cat((x_plus_position, x_plus_velocity, x_acceleration), dim=1)

        return x_plus

    def loss_function(self, predictions, targets):
        targets = targets.squeeze(1)
        return F.mse_loss(predictions, targets)

class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MultiLayerPerceptron, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.flatten(start_dim=1)
        x = self.layers(x)
        x = x.view(batch_size, -1, self.output_dim)
        return x

    #Febin
    def loss_function(self, predictions, targets):
        #print("inside mlp loss fn")

        # Ensure predictions and targets have the same shape
        #targets = targets.view_as(predictions) #Febin


        return F.mse_loss(predictions, targets)


