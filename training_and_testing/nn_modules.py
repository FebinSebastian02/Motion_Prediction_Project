import torch
from torch.nn import functional as F
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# TODO: Here you should create your models.
#  Each model should have a __init__ function, a forward function, and a loss_function function.
#  The loss function doesn't have to be in the model, but it is convenient to have it there, because the lit_module
#  will call it automatically, because you assign a prediction model to it and later it asks the model for the loss function.
class ConstantVelocityModel(nn.Module):
    def __init__(self, dt=1.0):
        super(ConstantVelocityModel, self).__init__()
        self.dt = dt

        # For plotting and printing ground truths and predictions later
        self.target_values = []
        self.prediction_values = []

    def forward(self, x):
        x = x[:, -1, :]  # Take the last element in the sequence
        x_position = x[:, :2]  # First two columns for positions
        x_velocity = x[:, 2:]  # Last two columns for velocities

        # Update positions based on velocities and time step
        x_plus_position = x_position + self.dt * x_velocity

        # Combine updated positions and original velocities
        x_plus = torch.cat((x_plus_position, x_velocity),
                           dim=1)  # Concatenate updated positions and original velocities

        return x_plus

    def loss_function(self, predictions, targets):
        targets = targets.squeeze(1)  # To remove the extra dimension

        # Appending y_view and y_hat values to the lists to use it for printing and plotting ground truths and predictions later
        self.target_values.append(targets.detach().cpu().numpy().tolist())
        self.prediction_values.append(predictions.detach().cpu().numpy().tolist())

        return F.mse_loss(predictions, targets)

    def plot_results(self):
        # Flatten the lists
        targets_flat = [item for sublist in self.target_values for item in sublist]
        predictions_flat = [item for sublist in self.prediction_values for item in sublist]

        # Create a DataFrame and save to CSV
        df = pd.DataFrame({
            'x_ground_truth': [item[0] for item in targets_flat],
            'y_ground_truth': [item[1] for item in targets_flat],
            'x_predicted': [item[0] for item in predictions_flat],
            'y_predicted': [item[1] for item in predictions_flat]
        })

        file_name_csv = 'ConstantVelocity_Predictions.csv'
        file_path = 'C:\\Febin\\@RPTU\\Sem 2\\Seminar Electromobility\\Motion_Prediction\\code\\code\\test_loss_results'
        full_path = os.path.join(file_path, file_name_csv)

        # Save the DataFrame to the specified location
        df.to_csv(full_path, index=False)
        print(f"Saved predictions to {full_path}")

        df_subset = df.iloc[:1000]

        # --- Plot 1: X Ground Truth vs X Predicted ---
        fig1, ax1 = plt.subplots(figsize=(12, 8))  # Create a new figure window
        ax1.plot(df_subset['x_ground_truth'], label='X Ground Truth', color='blue', linestyle='-')
        ax1.plot(df_subset['x_predicted'], label='X Predicted', color='red', linestyle='--')
        ax1.set_title('X Ground Truth vs X Predicted')
        ax1.legend()

        # Show and save the X plot
        plt.tight_layout()
        file_name_plot_x = 'ConstantVelocity_X_Predictions_Plot.png'
        full_path_plot_x = os.path.join(file_path, file_name_plot_x)
        fig1.savefig(full_path_plot_x)
        print(f"Saved X plot to {full_path_plot_x}")
        plt.show()

        # --- Plot 2: Y Ground Truth vs Y Predicted ---
        fig2, ax2 = plt.subplots(figsize=(12, 8))  # Create another new figure window
        ax2.plot(df_subset['y_ground_truth'], label='Y Ground Truth', color='blue', linestyle='-')
        ax2.plot(df_subset['y_predicted'], label='Y Predicted', color='red', linestyle='--')
        ax2.set_title('Y Ground Truth vs Y Predicted')
        ax2.legend()

        # Show and save the Y plot
        plt.tight_layout()
        file_name_plot_y = 'ConstantVelocity_Y_Predictions_Plot.png'
        full_path_plot_y = os.path.join(file_path, file_name_plot_y)
        fig2.savefig(full_path_plot_y)
        print(f"Saved Y plot to {full_path_plot_y}")
        plt.show()

class ConstantAccelerationModel(nn.Module):
    def __init__(self, dt=1.0):
        super(ConstantAccelerationModel, self).__init__()
        self.dt = dt  # Time step

        # For plotting and printing ground truths and predictions later
        self.target_values = []
        self.prediction_values = []

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

        # Appending y_view and y_hat values to the lists to use it for printing and plotting ground truths and predictions later
        self.target_values.append(targets.detach().cpu().numpy().tolist())
        self.prediction_values.append(predictions.detach().cpu().numpy().tolist())

        return F.mse_loss(predictions, targets)

    def plot_results(self):
        # Flatten the lists
        targets_flat = [item for sublist in self.target_values for item in sublist]
        predictions_flat = [item for sublist in self.prediction_values for item in sublist]

        # Create a DataFrame and save to CSV
        df = pd.DataFrame({
            'x_ground_truth': [item[0] for item in targets_flat],
            'y_ground_truth': [item[1] for item in targets_flat],
            'x_predicted': [item[0] for item in predictions_flat],
            'y_predicted': [item[1] for item in predictions_flat]
        })

        file_name = 'ConstantAcceleration_Predictions.csv'
        file_path = 'C:\\Febin\\@RPTU\\Sem 2\\Seminar Electromobility\\Motion_Prediction\\code\\code\\test_loss_results'
        full_path = os.path.join(file_path, file_name)

        # Save the DataFrame to the specified location
        df.to_csv(full_path, index=False)
        print(f"Saved predictions to {full_path}")

class BicycleModel(nn.Module):
    def __init__(self, L=2.5, dt=1):
        super(BicycleModel, self).__init__()
        self.dt = dt  # Timestep
        self.L = L  # Wheelbase

        # For plotting and printing ground truths and predictions later
        self.target_values = []
        self.prediction_values = []

    def forward(self, x):
        x = x[:, -1, :]  # Take the last element in the sequence

        x_pos = x[:, 0:1]  # xCenter
        y_pos = x[:, 1:2]  # yCenter
        heading = x[:, 2:3]
        xv = x[:, 3:4]  # xVelocity
        yv = x[:, 4:5]  # yVelocity
        xa = x[:, 5:6]  # xAcceleration
        ya = x[:, 6:7]  # yAcceleration
        lon_velocity = x[:, 7:8]  # lonVelocity
        lat_velocity = x[:, 8:9]  # latVelocity
        lon_acceleration = x[:, 9:10]  # lonAcceleration
        lat_acceleration = x[:, 10:11]  # latAcceleration

        # Compute state derivatives based on heading and velocity
        x_dot = xv * torch.cos(heading) - yv * torch.sin(heading)
        y_dot = xv * torch.sin(heading) + yv * torch.cos(heading)
        heading_dot = (xv * lat_velocity) / self.L  # Updated to reflect yaw rate dynamics

        xv_dot = xa  # longitudinal acceleration
        ya_dot = xv * heading_dot  # centripetal acceleration

        # Update state
        x_pos = x_pos + x_dot * self.dt
        y_pos = y_pos + y_dot * self.dt
        heading = heading + heading_dot * self.dt
        xv = xv + xv_dot * self.dt
        ya = ya + ya_dot * self.dt

        # Concatenate updated state
        x_plus = torch.cat(
            (x_pos, y_pos, heading, xv, yv, xa, ya, lon_velocity, lat_velocity, lon_acceleration, lat_acceleration),
            dim=1
        )

        return x_plus

    def loss_function(self, predictions, targets):
        targets = targets.squeeze(1)

        # Appending y_view and y_hat values to the lists to use it for printing and plotting ground truths and predictions later
        self.target_values.append(targets.detach().cpu().numpy().tolist())
        self.prediction_values.append(predictions.detach().cpu().numpy().tolist())

        return F.mse_loss(predictions, targets)

    def plot_results(self):
        # Flatten the lists
        targets_flat = [item for sublist in self.target_values for item in sublist]
        predictions_flat = [item for sublist in self.prediction_values for item in sublist]

        # Create a DataFrame and save to CSV
        df = pd.DataFrame({
            'x_ground_truth': [item[0] for item in targets_flat],
            'y_ground_truth': [item[1] for item in targets_flat],
            'x_predicted': [item[0] for item in predictions_flat],
            'y_predicted': [item[1] for item in predictions_flat]
        })

        file_name = 'Bicycle_Predictions.csv'
        file_path = 'C:\\Febin\\@RPTU\\Sem 2\\Seminar Electromobility\\Motion_Prediction\\code\\code\\test_loss_results'
        full_path = os.path.join(file_path, file_name)

        # Save the DataFrame to the specified location
        df.to_csv(full_path, index=False)
        print(f"Saved predictions to {full_path}")

        df_subset = df.iloc[:1000]

        # --- Plot 1: X Ground Truth vs X Predicted ---
        fig1, ax1 = plt.subplots(figsize=(12, 8))  # Create a new figure window
        ax1.plot(df_subset['x_ground_truth'], label='X Ground Truth', color='blue', linestyle='-')
        ax1.plot(df_subset['x_predicted'], label='X Predicted', color='red', linestyle='--')
        ax1.set_title('X Ground Truth vs X Predicted')
        ax1.legend()

        # Show and save the X plot
        plt.tight_layout()
        file_name_plot_x = 'Bicycle_X_Predictions_Plot.png'
        full_path_plot_x = os.path.join(file_path, file_name_plot_x)
        fig1.savefig(full_path_plot_x)
        print(f"Saved X plot to {full_path_plot_x}")
        plt.show()

        # --- Plot 2: Y Ground Truth vs Y Predicted ---
        fig2, ax2 = plt.subplots(figsize=(12, 8))  # Create another new figure window
        ax2.plot(df_subset['y_ground_truth'], label='Y Ground Truth', color='blue', linestyle='-')
        ax2.plot(df_subset['y_predicted'], label='Y Predicted', color='red', linestyle='--')
        ax2.set_title('Y Ground Truth vs Y Predicted')
        ax2.legend()

        # Show and save the Y plot
        plt.tight_layout()
        file_name_plot_y = 'Bicycle_Y_Predictions_Plot.png'
        full_path_plot_y = os.path.join(file_path, file_name_plot_y)
        fig2.savefig(full_path_plot_y)
        print(f"Saved Y plot to {full_path_plot_y}")
        plt.show()

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

        # For plotting and printing ground truths and predictions later
        self.target_values = []
        self.prediction_values = []

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.flatten(start_dim=1)
        x = self.layers(x)
        x = x.view(batch_size, -1, self.output_dim)
        return x

    def loss_function(self, predictions, targets):
        # Convert tensors to float32 before converting to NumPy arrays and lists
        self.target_values.append(targets.detach().cpu().to(torch.float32).numpy().tolist())
        self.prediction_values.append(predictions.detach().cpu().to(torch.float32).numpy().tolist())

        return F.mse_loss(predictions, targets)

    def plot_results(self):
        # Flatten the lists
        targets_flat = [item for sublist in self.target_values for item in sublist]
        predictions_flat = [item for sublist in self.prediction_values for item in sublist]

        # Create a DataFrame and save to CSV
        df = pd.DataFrame({
            'x_ground_truth': [item[0] for item in targets_flat],
            'y_ground_truth': [item[1] for item in targets_flat],
            'x_predicted': [item[0] for item in predictions_flat],
            'y_predicted': [item[1] for item in predictions_flat]
        })

        file_name = 'MLP_Predictions.csv'
        file_path = 'C:\\Febin\\@RPTU\\Sem 2\\Seminar Electromobility\\Motion_Prediction\\code\\code\\test_loss_results'
        full_path = os.path.join(file_path, file_name)

        # Save the DataFrame to the specified location
        df.to_csv(full_path, index=False)
        print(f"Saved predictions to {full_path}")

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(LSTMModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

        # For plotting and printing ground truths and predictions later
        self.target_values = []
        self.prediction_values = []

    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])

        return out.view(batch_size, -1, self.output_dim)

    def loss_function(self, predictions, targets):
        self.target_values.append(targets.detach().cpu().to(torch.float32).numpy().tolist())
        self.prediction_values.append(predictions.detach().cpu().to(torch.float32).numpy().tolist())

        return F.mse_loss(predictions, targets)

    def plot_results(self):
        # Flatten the lists
        targets_flat = [item for sublist in self.target_values for item in sublist]
        predictions_flat = [item for sublist in self.prediction_values for item in sublist]

        # Create a DataFrame and save to CSV
        df = pd.DataFrame({
            'x_ground_truth': [item[0] for item in targets_flat],
            'y_ground_truth': [item[1] for item in targets_flat],
            'x_predicted': [item[0] for item in predictions_flat],
            'y_predicted': [item[1] for item in predictions_flat]
        })

        file_name = 'LSTM_Predictions.csv'
        file_path = 'C:\\Febin\\@RPTU\\Sem 2\\Seminar Electromobility\\Motion_Prediction\\code\\code\\test_loss_results'
        full_path = os.path.join(file_path, file_name)

        # Save the DataFrame to the specified location
        df.to_csv(full_path, index=False)
        print(f"Saved predictions to {full_path}")


class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(GRU, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)  # Define the GRU layer
        self.fc = nn.Linear(hidden_dim, output_dim)  # Output layer

        # For plotting and printing ground truths and predictions later
        self.target_values = []
        self.prediction_values = []

    def forward(self, x):
        # Initialize hidden state for the first input
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(
            x.device)

        # Forward propagate the GRU
        out, _ = self.gru(x, h0)

        # Take the output from the last time step in the sequence
        out = out[:, -1, :]

        # Pass through the fully connected layer to get final predictions
        out = self.fc(out)

        return out

    def loss_function(self, predictions, targets):
        self.target_values.append(targets.detach().cpu().to(torch.float32).numpy().tolist())
        self.prediction_values.append(predictions.detach().cpu().to(torch.float32).numpy().tolist())

        return F.mse_loss(predictions, targets)

    def plot_results(self):
        # Flatten the lists
        targets_flat = [item for sublist in self.target_values for item in sublist]
        predictions_flat = [item for sublist in self.prediction_values for item in sublist]

        # Create a DataFrame and save to CSV
        df = pd.DataFrame({
            'x_ground_truth': [item[0] for item in targets_flat],
            'y_ground_truth': [item[1] for item in targets_flat],
            'x_predicted': [item[0] for item in predictions_flat],
            'y_predicted': [item[1] for item in predictions_flat]
        })

        file_name = 'GRU_Predictions.csv'
        file_path = 'C:\\Febin\\@RPTU\\Sem 2\\Seminar Electromobility\\Motion_Prediction\\code\\code\\test_loss_results'
        full_path = os.path.join(file_path, file_name)

        # Save the DataFrame to the specified location
        df.to_csv(full_path, index=False)
        print(f"Saved predictions to {full_path}")
