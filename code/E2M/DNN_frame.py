import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import TensorDataset, DataLoader 
import numpy as np
import pandas as pd
import os 
import random
import math
from sklearn.preprocessing import StandardScaler 
# Import Sparsemax
from sparsemax import Sparsemax
# import osqp
# from scipy import sparse
# class self:
#   pass

# Define a normalization layer with learnable parameters
class Norm_E2M(nn.Module):
    def __init__(self, d, axis = -2, eps = 1e-6):
        super().__init__()
        self.d = d  # Dimensionality of the input
        self.axis = axis  # Axis along which normalization is performed
        self.eps = eps  # Small constant to prevent division by zero
    
        # Learnable parameters for scaling (alpha) and shifting (bias)
        if axis == -2: # If normalizing by columns (Batch Norm style)
            self.alpha = nn.Parameter(torch.randn(d)) # Scaling parameter
            self.bias = nn.Parameter(torch.randn(d)) # Bias parameter
        else: # If normalizing by rows (Layer Norm style)
            self.alpha = nn.Parameter(torch.ones(d)) # Scaling parameter
            self.bias = nn. Parameter(torch.zeros(d)) # Bias parameter
    
    def forward(self,x):
        # Get the size of the dimension we are normalizing over
        dim_size = x.shape[self.axis]

        # Compute mean and standard deviation along the specified axis
        if dim_size > 1:
            avg = x.mean(axis=self.axis, keepdim=True)
            std = x.std(axis=self.axis, unbiased=False, keepdim=True) + self.eps
        else:
            avg = x
            std = torch.full_like(x, self.eps)

        # Apply normalization
        norm = self.alpha * (x - avg) / std + self.bias
        return norm

# Simplified MLP using Softmax weights and supporting Entropy Penalty
class MLP(nn.Module):
    def __init__(self, model_settings):
        super(MLP, self).__init__()
        # Core parameters from settings
        self.hidden = model_settings['hidden']
        self.layer = model_settings['number_layer'] - 1
        self.y_train = model_settings['y_train'] # Reference to training responses
        self.n = self.y_train.shape[0] # Number of training samples
        self.data_type = model_settings['data_type'] # Type of response data
        self.lamb = model_settings['lamb'] # Lambda for entropy penalty
        self.y_scale = model_settings['y_scale'] # Scale factor for MSE loss
        self.norm_axis = model_settings.get('norm_axis', -2) # Normalization axis
        self.num_features = model_settings['num_features'] # Number of input features
        self.dropout = model_settings['dropout'] # Dropout rate
        self.metric = model_settings['metric'] # Metric for SPD data

        # Remove temperature parameter
        # self.temperature = nn.Parameter(torch.tensor(1.0))

        # -- Network Architecture --
        self.linear_1 = nn.Linear(self.num_features, self.hidden)
        self.linear_hidden = nn.ModuleList()
        for i in range(self.layer):
            self.linear_hidden.append(nn.Linear(self.hidden, self.hidden))
        self.linear_out_l2 = nn.Linear(self.hidden, self.n) # Output layer (pre-weights)

        # Normalization layers
        self.norm1 = Norm_E2M(self.hidden, axis = self.norm_axis)
        self.linear_bn = nn.ModuleList()
        for i in range(self.layer):
            self.linear_bn.append(Norm_E2M(self.hidden, axis = self.norm_axis))
        self.norm_out = Norm_E2M(self.n, axis = self.norm_axis)

        # Dropout layers
        self.drop = nn.ModuleList()
        for i in range(self.layer):
            self.drop.append(nn.Dropout(self.dropout))
        self.drop_out = nn.Dropout(self.dropout)

        # Attribute to store weights for entropy penalty calculation
        self.linear_out_weight = None

    def forward(self, x):
        # Input layer
        x_ = F.relu(self.norm1(self.linear_1(x)))

        # Hidden layers with residual connections
        for i in range(self.layer):
            x_ = x_ + F.relu(self.linear_bn[i](self.linear_hidden[i](self.drop[i](x_))))

        # Final layer activation and normalization (outputting pre-weight logits)
        x_ = self.norm_out(self.linear_out_l2(self.drop_out(x_)))

        # Calculate weights using standard softmax
        weight = F.softmax(x_, dim=1)
        # Store weights for entropy penalty calculation in the loss function
        self.linear_out_weight = weight

        # Calculate final output by weighting training responses
        if self.data_type == "SPD" and self.metric == "BW":
            yout = BW_barycenter(weight, self.y_train)
        else:
            yout = torch.mm(weight, self.y_train)

        return yout

def bures_wasserstein_loss_batch(y_pred, y_obs, eps=1e-6):
    """
    Exact Bures-Wasserstein distance computation for SPD matrices.
    Formula: (tr(A) + tr(B) - 2 * tr((A^(1/2) * B * A^(1/2))^(1/2)))^(1/2)
    
    Args:
        y_pred: (batch_size, m*m) flattened SPD matrices
        y_obs: (batch_size, m*m) flattened SPD matrices  
        eps: small value for numerical stability
    
    Returns:
        bw_distances: (batch_size,) tensor of BW distances
    """
    batch_size = y_pred.shape[0]
    m = int(math.sqrt(y_pred.shape[1]))
    
    # Reshape to (batch_size, m, m)
    A_batch = y_pred.view(batch_size, m, m)
    B_batch = y_obs.view(batch_size, m, m)
    
    # Compute trace terms efficiently
    tr_A = torch.diagonal(A_batch, dim1=-2, dim2=-1).sum(-1)  # (batch_size,)
    tr_B = torch.diagonal(B_batch, dim1=-2, dim2=-1).sum(-1)  # (batch_size,)
    
    # Compute A^(1/2) using eigendecomposition
    # For SPD matrices: A = U * diag(λ) * U^T, so A^(1/2) = U * diag(√λ) * U^T
    eigenvals_A, eigenvecs_A = torch.linalg.eigh(A_batch)  # (batch_size, m), (batch_size, m, m)
    eigenvals_A = torch.clamp(eigenvals_A, min=eps)  # Ensure positive eigenvalues
    sqrt_eigenvals_A = torch.sqrt(eigenvals_A)  # (batch_size, m)
    sqrt_A = eigenvecs_A @ torch.diag_embed(sqrt_eigenvals_A) @ eigenvecs_A.transpose(-2, -1)
    
    # Compute A^(1/2) * B * A^(1/2)
    sqrt_A_B_sqrt_A = sqrt_A @ B_batch @ sqrt_A  # (batch_size, m, m)
    
    # Compute (A^(1/2) * B * A^(1/2))^(1/2)
    eigenvals_AB, eigenvecs_AB = torch.linalg.eigh(sqrt_A_B_sqrt_A)
    eigenvals_AB = torch.clamp(eigenvals_AB, min=eps)  # Ensure positive eigenvalues
    sqrt_eigenvals_AB = torch.sqrt(eigenvals_AB)  # (batch_size, m)
    
    # Compute tr((A^(1/2) * B * A^(1/2))^(1/2)) = sum of sqrt eigenvalues
    tr_sqrt_AB_sqrt = sqrt_eigenvals_AB.sum(-1)  # (batch_size,)
    
    # Exact Bures-Wasserstein distance: (tr(A) + tr(B) - 2*tr((A^(1/2)*B*A^(1/2))^(1/2)))^(1/2)
    bw_squared = tr_A + tr_B - 2 * tr_sqrt_AB_sqrt
    
    # Ensure non-negative before taking square root
    bw_squared = torch.clamp(bw_squared, min=eps)
    bw_distances = torch.sqrt(bw_squared)
    
    return bw_distances


def prediction_loss(y_pred, y_obs, model_settings):
    """Calculates only the prediction loss (Scaled MSE) for evaluation."""
    y_scale = model_settings.get('y_scale', 1.0)
    if isinstance(y_scale, torch.Tensor):
        y_scale = y_scale.item()
    y_scale = y_scale if y_scale != 0 else 1.0
    
    if model_settings['data_type'] == "SPD" and model_settings.get('metric') == "BW":
        # Efficient Bures-Wasserstein loss for SPD matrices
        bw_distances = bures_wasserstein_loss_batch(y_pred, y_obs)
        loss = bw_distances.mean()
    else:
        # Standard MSE loss for other data types
        loss = (y_pred - y_obs).pow(2).mean() / y_scale
    
    return loss

def prediction_loss_no_scale(y_pred, y_obs, model_settings):
    """Calculates only the prediction loss (Scaled MSE) for evaluation."""
    
    if model_settings['data_type'] == "SPD" and model_settings.get('metric') == "BW":
        # Efficient Bures-Wasserstein loss for SPD matrices
        bw_distances = bures_wasserstein_loss_batch(y_pred, y_obs)
        loss = bw_distances.mean()
    elif model_settings['data_type'] == "measure":
        # Standard MSE loss for other data types
        loss = (y_pred - y_obs).pow(2).mean(axis=1).mean()
    elif model_settings['data_type'] == "network":
        # Standard MSE loss for other data types
        loss = (y_pred - y_obs).pow(2).sum(axis=1).mean()
    elif model_settings['data_type'] == "SPD":
        # Standard MSE loss for other data types
        loss = (y_pred - y_obs).pow(2).sum(axis=1).mean()
    
    return loss

def custom_loss_DWR(y_pred, y_obs, model, model_settings):
    """Calculates the training loss (Scaled MSE + optional Entropy Penalty)."""
    lamb = model_settings.get('lamb', 0.0) # Get lambda for entropy penalty
    y_scale = model_settings.get('y_scale', 1.0)

    if isinstance(y_scale, torch.Tensor):
        y_scale = y_scale.item()
    y_scale = y_scale if y_scale != 0 else 1.0

    # Base loss: Use Bures-Wasserstein for SPD matrices, otherwise use MSE
    if model_settings['data_type'] == "SPD" and model_settings.get('metric') == "BW":
        # Use the same Bures-Wasserstein implementation as in prediction_loss
        base_loss = prediction_loss(y_pred, y_obs, model_settings) # Remove scaling, will be applied later
    else:
        # Standard scaled Mean Squared Error
        base_loss = (y_pred - y_obs).pow(2).mean()
    
    # Apply scaling
    mse_loss = base_loss / y_scale
    total_loss = mse_loss

    # Add entropy penalty if specified
    if lamb is not None and lamb != 0.0:
        if hasattr(model, 'linear_out_weight') and model.linear_out_weight is not None:
            weights = model.linear_out_weight
            # Add small epsilon inside log for stability
            entropy = -torch.sum(weights * torch.log(weights + 1e-10), dim=1).mean()
            # Note: Entropy is typically negative sum, so adding lamb * (-entropy)
            # Here we calculate sum(w*log(w)), which is negative entropy, so we add it directly.
            total_loss = mse_loss + lamb * entropy
        else:
            print("Warning: Entropy penalty requested but model.linear_out_weight not found or None.")

    return total_loss

def BW_barycenter(weights, spd_matrices, max_iter=10, eps=1e-6):
    """
    Compute Bures-Wasserstein barycenter of SPD matrices using the correct fixed-point iteration.
    
    The barycenter S minimizes: sum_i w_i * BW(S, A_i)^2
    where BW is the Bures-Wasserstein distance.
    
    Uses the fixed-point iteration from the paper:
    S^(k+1) = S^(k)^(-1/2) * (sum_i w_i * (S^(k)^(1/2) * A_i * S^(k)^(1/2))^(1/2))^2 * S^(k)^(-1/2)
    
    Args:
        weights: (batch_size, n) tensor of weights for each SPD matrix
        spd_matrices: (n, m*m) tensor of flattened SPD matrices
        max_iter: maximum number of iterations
        eps: small value for numerical stability
    
    Returns:
        barycenters: (batch_size, m*m) tensor of flattened barycenter matrices
    """
    batch_size, n = weights.shape
    m_squared = spd_matrices.shape[1]
    m = int(math.sqrt(m_squared))
    
    # Reshape SPD matrices to (n, m, m)
    A_matrices = spd_matrices.view(n, m, m)
    
    # Initialize barycenter S^(0) as identity matrix (as suggested in the paper)
    S_current = torch.eye(m, device=weights.device, dtype=weights.dtype).unsqueeze(0).repeat(batch_size, 1, 1)  # (batch_size, m, m)
    
    # Fixed-point iteration
    for iteration in range(max_iter):
        # Compute S^(k)^(1/2) and S^(k)^(-1/2) for current iterate
        eigenvals_S, eigenvecs_S = torch.linalg.eigh(S_current)  # (batch_size, m), (batch_size, m, m)
        eigenvals_S = torch.clamp(eigenvals_S, min=eps)
        sqrt_eigenvals_S = torch.sqrt(eigenvals_S)  # (batch_size, m)
        inv_sqrt_eigenvals_S = 1.0 / sqrt_eigenvals_S  # (batch_size, m)
        
        sqrt_S = eigenvecs_S @ torch.diag_embed(sqrt_eigenvals_S) @ eigenvecs_S.transpose(-2, -1)  # (batch_size, m, m)
        inv_sqrt_S = eigenvecs_S @ torch.diag_embed(inv_sqrt_eigenvals_S) @ eigenvecs_S.transpose(-2, -1)  # (batch_size, m, m)
        
        # Compute weighted sum: sum_i w_i * (S^(k)^(1/2) * A_i * S^(k)^(1/2))^(1/2)
        weighted_sum = torch.zeros_like(S_current)  # (batch_size, m, m)
        
        for i in range(n):
            # Get weight for matrix A_i
            w_i = weights[:, i].unsqueeze(-1).unsqueeze(-1)  # (batch_size, 1, 1)
            A_i = A_matrices[i].unsqueeze(0)  # (1, m, m)
            
            # Compute S^(k)^(1/2) * A_i * S^(k)^(1/2) for all batch elements
            sandwich = sqrt_S @ A_i @ sqrt_S  # (batch_size, m, m)
            
            # Compute (S^(k)^(1/2) * A_i * S^(k)^(1/2))^(1/2)
            eigenvals_sandwich, eigenvecs_sandwich = torch.linalg.eigh(sandwich)  # (batch_size, m), (batch_size, m, m)
            eigenvals_sandwich = torch.clamp(eigenvals_sandwich, min=eps)
            sqrt_eigenvals_sandwich = torch.sqrt(eigenvals_sandwich)  # (batch_size, m)
            sqrt_sandwich = eigenvecs_sandwich @ torch.diag_embed(sqrt_eigenvals_sandwich) @ eigenvecs_sandwich.transpose(-2, -1)  # (batch_size, m, m)
            
            # Add weighted contribution
            weighted_sum += w_i * sqrt_sandwich
        
        # Square the weighted sum: (sum_i w_i * (S^(k)^(1/2) * A_i * S^(k)^(1/2))^(1/2))^2
        squared_sum = weighted_sum @ weighted_sum  # (batch_size, m, m)
        
        # Apply the transformation: S^(k+1) = S^(k)^(-1/2) * squared_sum * S^(k)^(-1/2)
        S_next = inv_sqrt_S @ squared_sum @ inv_sqrt_S  # (batch_size, m, m)
        
        # Check for convergence (optional, can be removed for fixed iterations)
        if iteration > 0:
            diff = torch.norm(S_next - S_current, dim=(-2, -1)).max()
            if diff < eps:
                break
        
        S_current = S_next
    
    # Flatten the result back to (batch_size, m*m)
    barycenters = S_current.view(batch_size, m_squared)
    
    return barycenters
