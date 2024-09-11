import time
import lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from utils import build_module
import torch.nn as nn


# TODO: Here you should create your models. You can use the MLPModel or ConstantVelocity as a template.
#  Each model should have a __init__ function, a forward function, and a loss_function function.
#  The loss function doen't have to be in the model, but it is convenient to have it there, because the lit_module
#  will call it automatically, because you assign a prediction model to it and later it asks the model for the loss function.
class ConstantVelocityModel(nn.Module):
    def __init__(self, dt=1.0):
        super(ConstantVelocityModel, self).__init__()
        self.dt = dt

    def forward(self, x):
        # print(f"Value of x1:- {x}")
        # time.sleep(10)
        x = x[:, -1, :]  ## Take the last element in the sequence
        # print(f"Value of x2:- {x}")
        # time.sleep(1)

        # Febin
        # The input tensor structure is [x_position, y_position, x_velocity, y_velocity]
        x_position = x[:, :2]  # First two columns for positions
        x_velocity = x[:, 2:]  # Last two columns for velocities

        # print(f"Value of x_position:- {x_position}")
        # time.sleep(10)

        # print(f"Value of x_velocity:- {x_velocity}")
        # time.sleep(1)

        # x_plus = x + self.dt * x

        # Febin
        # Update positions based on velocities and time step
        x_plus_position = x_position + self.dt * x_velocity
        # print(f"Value of updated position:- {x_plus_position}")
        # time.sleep(1)

        # Combine updated positions with original velocities
        x_plus = torch.cat((x_plus_position, x_velocity),
                           dim=1)  # velocity is constant and positions gets updated. Needed to avoid tensor size mismatch error too
        # print(f"Value of concatenated variable:- {x_plus}")
        # time.sleep(1)
        return x_plus

    # Febin
    def loss_function(self, predictions, targets):
        # print("inside constant velocity loss fn")

        # Ensure predictions and targets have the same shape
        # targets = targets.view_as(predictions) #Febin
        targets = targets.squeeze(1)  # Febin - To change shape from [43, 1, 12] to [43,12]
        """print(f"Shape of targets:- {targets.shape}")
        print(f"Predicted value: {predictions}")
        print(f"Target value: {targets}")"""
        return F.mse_loss(predictions, targets)

#Bicycle model
class BicycleModel(nn.Module):
    def __init__(self, L = 2.5, dt=1):#Listhedefaultwheelbase
        super(BicycleModel,self).__init__()
        self.dt=dt#Timestep
        self.L=L#Wheelbase

    def forward(self,x):
        x=x[:,-1,:]#Takethelastelementinthesequence

        x_pos=x[:,0:1]#xCenter
        y_pos=x[:,1:2]#yCenter
        heading=x[:,2:3]#heading
        xv=x[:,3:4]#xVelocity
        yv=x[:,4:5]#yVelocity
        xa=x[:,5:6]#xAcceleration
        ya=x[:,6:7]#yAcceleration
        lon_velocity=x[:,7:8]#lonVelocity
        lat_velocity=x[:,8:9]#latVelocity
        lon_acceleration=x[:,9:10]#lonAcceleration
        lat_acceleration=x[:,10:11]#latAcceleration

        #Computederivatives
        x_dot=xv*torch.cos(heading)
        y_dot=xv*torch.sin(heading)
        heading_dot=(xv*torch.tan(lat_velocity))/self.L
        #heading_dot=yv/xv#
        xv_dot=xa
        ya_dot=(((2*xa*lat_velocity)+(xv*lat_acceleration)))*xv/self.L
        xa_dot=lon_acceleration
        delta_dot=lat_acceleration

        #Updatestate
        x_pos=x_pos+x_dot*self.dt
        y_pos=y_pos+y_dot*self.dt
        heading=heading+heading_dot*self.dt
        xv=xv+xv_dot*self.dt
        ya=ya+ya_dot*self.dt
        xa=xa+xa_dot*self.dt
        lat_velocity=lat_velocity+delta_dot*self.dt

        #Concatenateupdatedstate
        x_plus=torch.cat(
        (x_pos,y_pos,heading,xv,yv,xa,ya,lon_velocity,lat_velocity,lon_acceleration,lat_acceleration),
        dim=1)

        return x_plus

    def loss_function(self, predictions, targets):
        """
        Loss function for the kinematic bicycle model (Mean Squared Error).
        :param predictions: Predicted future positions and velocities
        :param targets: Target future positions and velocities
        :return: MSE loss
        """
        # Ensure targets match predictions shape
        targets = targets.squeeze(1)
        return F.mse_loss(predictions, targets)


class ConstantAccelerationModel(nn.Module):
    def __init__(self, dt=1.0):
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

    # Febin
    def loss_function(self, predictions, targets):
        # print("inside mlp loss fn")
        # Ensure predictions and targets have the same shape
        # targets = targets.view_as(predictions) #Febin
        return F.mse_loss(predictions, targets)


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(LSTMModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)

        #print(f"x shape: {x.shape}")
        #print(f"h0 shape: {h0.shape}")
        #print(f"c0 shape: {c0.shape}")
        #x = x.contiguous()
        #print(f"Input is contiguous: {x.is_contiguous()}")

        out, _ = self.lstm(x, (h0, c0))
        #print(f"out shape: {out.shape}")

        out = self.fc(out[:, -1, :])

        return out.view(batch_size, -1, self.output_dim)

    def loss_function(self, predictions, targets):
        return F.mse_loss(predictions, targets)
