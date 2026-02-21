import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import os
import random
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold # Added for K-Fold CV
import itertools # Added for parameter combinations
import FNN_frame # Assuming FNN_frame.py is in the same directory or accessible path

# Define the deep neural network for regression
def FNN(x, y, xout, layer=[4], hidden=[64], dropout=[0.3], lr=[0.0005], num_epochs=2000, seed=None, batch_size=32, norm_axis=-2):

    # Validate norm_axis
    try:
        norm_axis = int(norm_axis)
        if norm_axis not in [-1, -2]:
            raise ValueError("norm_axis must be -1 or -2.")
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid value for norm_axis: {norm_axis}. Error: {e}")

    # --- Reproducibility Setup ---
    py_seed = None
    if seed is not None:
        try:
            py_seed = int(seed)
            random.seed(py_seed)
            torch.manual_seed(py_seed)
            np.random.seed(py_seed)
        except (ValueError, TypeError):
             print(f"Warning: Invalid seed value provided ({seed}). Proceeding without setting seed.")

    # --- Data Type Conversion and Preparation ---
    try:
        num_epochs = int(num_epochs)
        batch_size = int(batch_size)
    except (ValueError, TypeError) as e:
        raise ValueError(f"num_epochs and batch_size must be convertible to integers. Error: {e}")

    x_tensor = torch.tensor(np.array(x).astype(np.float32))
    y_tensor = torch.tensor(np.array(y).astype(np.float32))
    xout_tensor = None
    if xout is not None:
        xout_tensor = torch.tensor(np.array(xout).astype(np.float32))

    y_scale = y_tensor.mean(axis=0).pow(2).mean()
    n, p = x_tensor.shape

    # --- Hyperparameter Tuning Setup --- 
    # Define parameters available for tuning
    params_to_tune = {
        'layer': [int(l) for l in layer] if isinstance(layer, list) else [int(layer)],
        'hidden': [int(h) for h in hidden] if isinstance(hidden, list) else [int(hidden)],
        'dropout': [float(d) for d in dropout] if isinstance(dropout, list) else [float(dropout)],
        'lr': [float(l) for l in lr] if isinstance(lr, list) else [float(lr)],
    }

    cv_needed = any(len(v) > 1 for v in params_to_tune.values())
    best_params = {}
    param_keys = list(params_to_tune.keys())
    param_values = list(params_to_tune.values())

    if cv_needed:
        print("Starting 5-Fold Cross-Validation for hyperparameter tuning...")
        kf = KFold(n_splits=5, shuffle=True, random_state=py_seed)
        results = []
        param_combinations = list(itertools.product(*param_values))
        min_avg_cv_loss = float('inf')
        best_params_combo = None

        for i, combo in enumerate(param_combinations):
            current_params = dict(zip(param_keys, combo))
            fold_losses = []
            print(f"Testing Combination {i+1}/{len(param_combinations)}: {current_params}")

            for fold, (train_idx, val_idx) in enumerate(kf.split(x_tensor)):
                # ... (Inner CV loop setup remains the same) ...
                x_fold_train, y_fold_train = x_tensor[train_idx], y_tensor[train_idx]
                x_fold_val, y_fold_val = x_tensor[val_idx], y_tensor[val_idx]
                n_fold_train = len(train_idx)
                inner_indices = np.random.permutation(n_fold_train)
                inner_split_index = int(n_fold_train * 0.8)
                inner_train_indices = inner_indices[:inner_split_index]
                inner_val_indices = inner_indices[inner_split_index:]
                x_inner_train = x_fold_train[inner_train_indices,]
                y_inner_train = y_fold_train[inner_train_indices,]
                x_inner_valid = x_fold_train[inner_val_indices,]
                y_inner_valid = y_fold_train[inner_val_indices,]
                inner_train_ds = TensorDataset(x_inner_train, y_inner_train)
                batch_size_cv = min(batch_size, len(inner_train_indices))
                if batch_size_cv == 0:
                    fold_losses.append(float('inf'))
                    continue
                inner_train_dl = DataLoader(inner_train_ds, batch_size=batch_size_cv, shuffle=True)

                # Define model settings for this CV fold
                model_settings_cv = {
                    "num_features": p, "y_train": y_tensor, 
                    "number_layer": current_params['layer'], "hidden": current_params['hidden'],
                    "dropout": current_params['dropout'], "seed": py_seed,
                    "y_scale": y_scale, "norm_axis": norm_axis
                }

                model_cv = FNN_frame.FNN_MLP(model_settings_cv)
                optimizer_cv = torch.optim.Adam(model_cv.parameters(), lr=current_params['lr'])

                min_inner_valid_loss = float("inf")
                best_model_cv_state = None
                min_epoch_cv = 0

                # Inner training loop
                for epoch in range(num_epochs):
                    model_cv.train()
                    cost = None
                    for batch_idx, (x_obs, y_obs) in enumerate(inner_train_dl):
                        y_pred = model_cv(x_obs)
                        # Loss is now a direct call to scaled_mse_loss
                        cost = FNN_frame.scaled_mse_loss(y_pred, y_obs, model_settings_cv)
                        if torch.isnan(cost) or torch.isinf(cost):
                            break
                        optimizer_cv.zero_grad()
                        cost.backward()
                        optimizer_cv.step()
                    if cost is not None and (torch.isnan(cost) or torch.isinf(cost)):
                         break # Exit epoch loop
                    # Inner validation for early stopping
                    model_cv.eval()
                    with torch.no_grad():
                        y_pred_inner_val = model_cv(x_inner_valid)
                        # Use prediction loss (no penalty) for CV evaluation
                        cost_inner_val = FNN_frame.scaled_mse_loss(y_pred_inner_val, y_inner_valid, model_settings_cv)
                        inner_valid_loss = cost_inner_val.item()
                        if not (torch.isnan(cost_inner_val) or torch.isinf(cost_inner_val)):
                            if min_inner_valid_loss > inner_valid_loss:
                                min_inner_valid_loss = inner_valid_loss
                                best_model_cv_state = model_cv.state_dict()
                                min_epoch_cv = epoch
                            if min_epoch_cv + 200 < epoch:
                                break # Stop inner epoch loop

                # Evaluate best inner model on outer validation fold
                if best_model_cv_state:
                    model_cv.load_state_dict(best_model_cv_state)
                    model_cv.eval()
                    with torch.no_grad():
                        y_pred_fold_val = model_cv(x_fold_val)
                        cost_fold_val = FNN_frame.scaled_mse_loss(y_pred_fold_val, y_fold_val, model_settings_cv)
                        fold_loss_item = cost_fold_val.item()
                        if not (np.isnan(fold_loss_item) or np.isinf(fold_loss_item)):
                            fold_losses.append(fold_loss_item)
                        else:
                             fold_losses.append(float('inf'))
                else:
                    fold_losses.append(float('inf'))

            # Calculate average CV loss for the combination
            valid_fold_losses = [loss for loss in fold_losses if not np.isinf(loss)]
            avg_cv_loss = np.mean(valid_fold_losses) if valid_fold_losses else float('inf')
            print(f"  Avg CV Loss for Combo {i+1}: {avg_cv_loss:.6f}")
            results.append({'params': current_params, 'avg_cv_loss': avg_cv_loss})
            if avg_cv_loss < min_avg_cv_loss:
                min_avg_cv_loss = avg_cv_loss
                best_params_combo = current_params

        # Select best parameters overall
        if best_params_combo:
            best_params = best_params_combo
            print(f"Best parameters found: {best_params}")
            print(f"Best average CV loss: {min_avg_cv_loss:.6f}")
        else:
             print("Cross-validation failed. Using initial defaults.")
             best_params = dict(zip(param_keys, [v[0] for v in param_values]))

    else: # No CV needed
        best_params = dict(zip(param_keys, [v[0] for v in param_values]))
        print(f"Skipping Cross-Validation. Using parameters: {best_params}")

    # --- Final Training using Best Parameters ---
    print("Starting final training with selected parameters...")
    # Define model settings for final training
    model_settings = {
        "num_features": p, "y_train": y_tensor, 
        "number_layer": best_params['layer'], "hidden": best_params['hidden'],
        "dropout": best_params['dropout'], "seed": py_seed,
        "y_scale": y_scale, "norm_axis": norm_axis
    }

    # ... (Final training loop setup remains the same) ...
    indices = np.random.permutation(n)
    split_index = int(n * 0.8)
    train_indices = indices[:split_index]
    val_indices = indices[split_index:]
    x_train_final = x_tensor[train_indices,]
    y_train_final = y_tensor[train_indices,]
    x_valid_final = x_tensor[val_indices,]
    y_valid_final = y_tensor[val_indices,]
    train_ds_final = TensorDataset(x_train_final, y_train_final)
    batch_size_final = min(batch_size, len(x_train_final))
    if batch_size_final <= 0:
         raise ValueError(f"Final training batch size <= 0.")
    train_dl_final = DataLoader(train_ds_final, batch_size=batch_size_final, shuffle=True)
    final_model = FNN_frame.FNN_MLP(model_settings)
    final_optimizer = torch.optim.Adam(final_model.parameters(), lr=best_params['lr'])
    min_valid_loss_final = float("inf")
    best_model_final_state = None
    min_epoch_final = 0
    final_err_train = []
    final_err_valid = []

    # Final training loop
    for epoch in range(num_epochs):
        final_model.train()
        epoch_train_loss = 0.0
        num_batches = 0
        cost = None
        for batch_idx, (x_obs, y_obs) in enumerate(train_dl_final):
            y_pred = final_model(x_obs)
            # Use scaled_mse_loss for training
            cost = FNN_frame.scaled_mse_loss(y_pred, y_obs, model_settings)
            if torch.isnan(cost) or torch.isinf(cost):
                break
            epoch_train_loss += cost.item()
            num_batches += 1
            final_optimizer.zero_grad()
            cost.backward()
            final_optimizer.step()
        if cost is not None and (torch.isnan(cost) or torch.isinf(cost)):
             break # Exit epoch loop
        avg_epoch_train_loss = epoch_train_loss / num_batches if num_batches > 0 else 0
        final_err_train.append(avg_epoch_train_loss)

        # Validation step
        final_model.eval()
        with torch.no_grad():
            y_pred_valid = final_model(x_valid_final)
            cost_valid = FNN_frame.scaled_mse_loss(y_pred_valid, y_valid_final, model_settings)
            valid_loss = cost_valid.item()
            final_err_valid.append(valid_loss)
            if not (torch.isnan(cost_valid) or torch.isinf(cost_valid)):
                if min_valid_loss_final > valid_loss:
                    min_valid_loss_final = valid_loss
                    best_model_final_state = final_model.state_dict()
                    min_epoch_final = epoch
            # if (epoch + 1) % 100 == 0:
            #      print(f"  Epoch {epoch+1}/{num_epochs}: Training Loss = {avg_epoch_train_loss:.4f}, Validation Loss = {valid_loss:.4f}")
            if min_epoch_final + 500 < epoch:
                print(f"  Final training early stopping at epoch {epoch+1}")
                break

    print("Final training finished.")
    # Load the best model state
    if best_model_final_state:
        final_model = FNN_frame.FNN_MLP(model_settings)
        final_model.load_state_dict(best_model_final_state)
        final_model.eval()
        # print(f"Loaded best model from epoch {min_epoch_final + 1}.")
    else:
        print("Warning: No valid final model state found.")
        final_model.eval()

    # --- Prepare Results --- 
    # Remove learned_temperature from return dict
    return_dict = {
        "best_params": best_params if cv_needed else None,
        "final_err_train": final_err_train,
        "final_err_valid": final_err_valid,
        "min_final_valid_loss": min_valid_loss_final
    }

    # --- Final Predictions --- 
    if best_model_final_state:
        with torch.no_grad():
            y_fit_final = final_model(x_tensor)
            # Squeeze the output to remove the trailing dimension (n, 1) -> (n,)
            return_dict["y_fit"] = y_fit_final.squeeze(-1).detach().numpy()
            if xout_tensor is not None:
                # Single sample prediction check remains relevant for norm_axis
                if xout_tensor.shape[0] == 1 and model_settings.get('norm_axis', -2) != -1:
                    print("Warning: Switching norm_axis to -1 for single-sample prediction.")
                    prediction_settings = model_settings.copy()
                    prediction_settings['norm_axis'] = -1
                    pred_model = FNN_frame.FNN_MLP(prediction_settings)
                    pred_model.load_state_dict(final_model.state_dict())
                    pred_model.eval()
                    y_pred_final = pred_model(xout_tensor)
                else:
                    y_pred_final = final_model(xout_tensor)
                # Squeeze the output to remove the trailing dimension
                return_dict["y_pred"] = y_pred_final.squeeze(-1).detach().numpy()
    else:
        print("Warning: Skipping final predictions as no valid model found.")
        y_fit_shape = (n, y_tensor.shape[1]) if len(y_tensor.shape) > 1 else (n,)
        return_dict["y_fit"] = np.full(y_fit_shape, np.nan)
        if xout_tensor is not None:
            y_pred_shape = (xout_tensor.shape[0], y_tensor.shape[1]) if len(y_tensor.shape) > 1 else (xout_tensor.shape[0],)
            return_dict["y_pred"] = np.full(y_pred_shape, np.nan)

    return return_dict
  
