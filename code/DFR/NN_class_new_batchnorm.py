import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import TensorDataset, DataLoader 
import numpy as np
import pandas as pd
import os 
import random
from sklearn.preprocessing import StandardScaler 

# Define a normalization layer with learnable parameters
class Norm_DFR(nn.Module):
  def __init__(self, d, axis = -2, eps = 1e-6):
    super().__init__()
    self.d = d  # Dimensionality of the input
    self.axis = axis  # Axis along which normalization is performed
    self.eps = eps  # Small constant to prevent division by zero
    
    # Initialize learnable parameters for scaling and shifting
    if axis == -2: # If normalizing by columns
      self.alpha = nn.Parameter(torch.randn(d)) # Scaling parameter
      self.bias = nn.Parameter(torch.randn(d)) # Bias parameter
    else: # If normalizing by rows
      self.alpha = nn.Parameter(torch.randn(d)) # Scaling parameter
      self.bias = nn. Parameter(torch.randn(d)) # Bias parameter
    
  def forward(self,x):
    # Compute mean and standard deviation along the specified axis
    avg = x.mean(axis=self.axis, keepdim=True) if self.d > 1 else 0
    std = x.std(axis=self.axis, keepdim=True) + self.eps if self.d > 1 else 1
    # Normalize input using learnable parameters
    norm = self.alpha * (x - avg) / std + self.bias
    return norm

# Define a Multilayer Perceptron (MLP) model
class MultilayerPerceptron(nn.Module):
  def __init__(self, num_features, num_response, number_layer=4, hidden=64, dropout=0.3):
    super(MultilayerPerceptron, self).__init__()
    self.hidden = hidden  # Number of neurons in hidden layers
    self.layer = number_layer - 1  # Number of hidden layers (excluding input and output)
    
    # Input layer
    self.linear_1 = nn.Linear(num_features, self.hidden)
    
    # Hidden layers (stored in a ModuleList for flexibility)
    self.linear_hidden = nn.ModuleList()
    for i in range(self.layer):
      self.linear_hidden.append(nn.Linear(self.hidden, self.hidden))
    
    # Output layer
    self.linear_out = nn.Linear(self.hidden, num_response)
    
    # Normalization layers
    self.norm1 = Norm_DFR(self.hidden, axis = -1) # Normalization after the first layer
    self.linear_bn = nn.ModuleList()
    for i in range(self.layer):
      self.linear_bn.append(Norm_DFR(self.hidden, axis = -1)) # Normalization after hidden layers
    self.norm_out = Norm_DFR(num_response, axis = -1) # Normalization after the output layer
    
    # Dropout layers (to prevent overfitting)
    self.drop = nn.ModuleList()
    for i in range(self.layer):
      self.drop.append(nn.Dropout(dropout)) # Dropout applied before each hidden layer


  def forward(self, x):
    # Pass input through the input layer and apply ReLU activation
    x_ = F.relu(self.linear_1(x))
    x_ = self.norm1(x_) # Apply normalization
    
    # Pass through each hidden layer with residual connections
    for i in range(self.layer):
      # Residual connection: add input to the output of the layer
      x_ = x_ + F.relu(self.linear_hidden[i](self.drop[i](x_)))
      x_ = self.linear_bn[i](x_) # Apply normalization
    
    # Pass through the output layer
    x_ = self.linear_out(x_)
    x_ = self.norm_out(x_)  # Apply normalization to the output
    
    return x_
