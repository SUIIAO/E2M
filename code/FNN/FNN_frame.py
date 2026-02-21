import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import TensorDataset, DataLoader 
import numpy as np
import pandas as pd
import os 
import random
from sklearn.preprocessing import StandardScaler 
# Import Sparsemax
from sparsemax import Sparsemax
# import osqp
# from scipy import sparse
# class self:
#   pass

# Define a normalization layer with learnable parameters
class Norm_FNN(nn.Module):
    def __init__(self, d, axis=-2, eps=1e-6):
        super().__init__()
        self.d = d
        self.axis = axis
        self.eps = eps
    
        # Learnable parameters for scaling (alpha) and shifting (bias)
        # For both styles, we initialize to achieve an identity transform initially, which is a stable default.
        if axis == -2: # Batch Norm style
            self.alpha = nn.Parameter(torch.ones(d)) # Scaling parameter
            self.bias = nn.Parameter(torch.zeros(d)) # Bias parameter
        else: # Layer Norm style
            self.alpha = nn.Parameter(torch.ones(d)) # Scaling parameter
            self.bias = nn.Parameter(torch.zeros(d)) # Bias parameter
    
    def forward(self,x):
        # Get the size of the dimension we are normalizing over
        dim_size = x.shape[self.axis]

        # Compute mean and standard deviation along the specified axis
        # Note: This does not use running stats for eval mode, which is why FNN.py
        # has a special case for single-sample prediction when using axis=-2.
        if dim_size > 1:
            avg = x.mean(axis=self.axis, keepdim=True)
            std = x.std(axis=self.axis, unbiased=False, keepdim=True) + self.eps
        else:
            avg = x
            std = torch.full_like(x, self.eps)

        # Apply normalization
        norm = self.alpha * (x - avg) / std + self.bias
        return norm

# A standard Feedforward Neural Network (FNN) for regression.
class FNN_MLP(nn.Module):
    def __init__(self, model_settings):
        super(FNN_MLP, self).__init__()
        # Core parameters from settings
        self.hidden = model_settings['hidden']
        num_layers = model_settings['number_layer']
        self.norm_axis = model_settings.get('norm_axis', -2)
        self.num_features = model_settings['num_features']
        self.dropout_rate = model_settings['dropout']

        # Build the network layers sequentially
        layers = []
        input_dim = self.num_features
        for i in range(num_layers):
            # For all but the last layer, the output dimension is the hidden dimension
            output_dim = self.hidden
            
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(Norm_FNN(output_dim, axis=self.norm_axis))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout_rate))
            
            # The next layer's input is the current layer's output
            input_dim = output_dim

        self.network = nn.Sequential(*layers)
        
        # Final output layer to produce a single scalar value
        self.output_layer = nn.Linear(self.hidden, 1)

    def forward(self, x):
        # Pass input through the main network
        x_out = self.network(x)
        # Pass the result through the final output layer
        yout = self.output_layer(x_out)
        return yout

def scaled_mse_loss(y_pred, y_obs, model_settings):
    """Calculates the prediction loss (Scaled MSE)."""
    y_scale = model_settings.get('y_scale', 1.0)
    if isinstance(y_scale, torch.Tensor):
        y_scale = y_scale.item()
    # Ensure y_scale is not zero to avoid division by zero
    y_scale = y_scale if y_scale != 0 else 1.0
    loss = (y_pred - y_obs).pow(2).mean() / y_scale
    return loss
