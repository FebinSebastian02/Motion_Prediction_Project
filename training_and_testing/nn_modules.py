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
        """if x.size(1) == 0:  #Febin
            raise ValueError("Input tensor has no time steps.")     #Febin"""
        x = x[:, -1, :]
        x_plus = x + self.dt * x
        return x_plus

    #Febin
    def loss_function(self, predictions, targets):
        #print("inside constant velocity loss fn")

        # Ensure predictions and targets have the same shape
        #targets = targets.view_as(predictions) #Febin

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