# Set the working directory to the current script location
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Library packages
library(pracma)
library(dplyr)
library(frechet)
library(fdadensity)
library(tidyverse)

library(foreach)
library(reshape2)

# Load functions for E2M model
source("../../code/E2M/E2M.R")
source("../../code/E2M/lcm.R")
reticulate::source_python("../../code/E2M/E2M.py")

# Load functions for GFR model
function_path = "../../code/Wasserstein-regression-with-empirical-measures/code/"
function_sources = list.files(function_path, pattern="*.R$", full.names=TRUE, 
                              ignore.case=TRUE)
sapply(function_sources, source, .GlobalEnv)

# Load functions for SDR model
function_path = "../../code/DR4FrechetReg/Functions"
function_sources = list.files(function_path, pattern="*.R$", full.names=TRUE, 
                              ignore.case=TRUE)
sapply(function_sources, source, .GlobalEnv)

# Load functions for DFR model and required Python scripts
function_path = "../../code/DFR"
function_sources = list.files(function_path, pattern="*.R$", full.names=TRUE, 
                              ignore.case=TRUE)
sapply(function_sources, source, .GlobalEnv)
reticulate::source_python('../../code/DFR/NN_class_new_batchnorm.py')
reticulate::source_python('../../code/DFR/DNN_dy_layers.py')

# Load functions for IFR model
source("../../code/Single-Index-Frechet/SIdxDenReg.R")

# Predictor
# Load the mortality data
mortality = readRDS("mortality.RData")

# Extract predictor matrix and density data
x_pred = mortality$pred

# Convert density data to quantile functions
quan = foreach(i = (1:nrow(x_pred)), .combine = "rbind") %do% {
  x = mortality$density[[i]]$x
  y = mortality$density[[i]]$y
  y.quantile = dens2quantile(dens = y, dSup = x)
}
n = nrow(x_pred)

# Set up parallel processing
library(doRNG)
library(doParallel)
ncores = detectCores() - 1
cl = makeCluster(ncores) 
length(cl)
registerDoParallel(cl)

result = foreach(q = 1:n, .packages = c('tidyverse', 'dplyr', 'frechet', 'foreach', 'reshape2', 'vegan', 'reticulate', 'pracma', 'fdadensity')) %dorng% {
  # Reload necessary functions for each iteration
  source("../../code/E2M/E2M.R")
  source("../../code/E2M/lcm.R")
  reticulate::source_python("../../code/E2M/E2M.py")
  function_path = "../../code/Wasserstein-regression-with-empirical-measures/code/"
  function_sources = list.files(function_path, pattern="*.R$", full.names=TRUE, 
                                ignore.case=TRUE)
  sapply(function_sources, source, .GlobalEnv)
  function_path = "../../code/DR4FrechetReg/Functions"
  function_sources = list.files(function_path, pattern="*.R$", full.names=TRUE, 
                                ignore.case=TRUE)
  sapply(function_sources, source, .GlobalEnv)
  function_path = "../../code/DFR"
  function_sources = list.files(function_path, pattern="*.R$", full.names=TRUE, 
                                ignore.case=TRUE)
  sapply(function_sources, source, .GlobalEnv)
  reticulate::source_python('../../code/DFR/NN_class_new_batchnorm.py')
  reticulate::source_python('../../code/DFR/DNN_dy_layers.py')
  source("../../code/Single-Index-Frechet/SIdxDenReg.R")
  
  ind_test = q
  ind_remain = setdiff(1:162, ind_test)
  
  quan_train = quan[ind_remain,]
  quan_test = matrix(quan[ind_test,], nrow = length(ind_test))
  x_pred_train = x_pred[ind_remain,]
  x_pred_test = matrix(x_pred[ind_test,], nrow = length(ind_test))
  
  x_pred_train_mean = as.vector(colMeans(x_pred_train))
  x_pred_train_sd = as.vector(apply(x_pred_train, 2, sd))
  x_pred_train = t((t(x_pred_train) - x_pred_train_mean)/x_pred_train_sd) ## standardize individually
  x_pred_test = t((t(x_pred_test) - x_pred_train_mean)/x_pred_train_sd)  ## standardize individually
  
  y = lapply(1:nrow(quan_train), function(j) quan_train[j,])
  
  ################################### E2M ######################################
  layer = c(2,3,4,5,6)
  hidden = c(8,16,32,64,128)
  lamb = c(-1e-1,-1e-2,0,1e-2,1e-1)
  res_E2M = E2M(y = y, x = x_pred_train, xout = x_pred_test,
                optns = list(type = "measure", metric = "wasserstein", 
                             layer = layer, hidden = hidden, lr = 0.0005,
                             lamb = lamb, num_epochs = 2000, dropout = 0.3, 
                             norm_axis = -1, seed = q)
  )
  err_E2M = sapply(1:length(ind_test), function(j) mean((res_E2M$yPred[j,] - quan_test[j,])^2))
  
  ##################################### DFR ####################################
  y = lapply(1:nrow(quan_train), function(j) quan_train[j, ])
  res_dfr = DFR(y = y, x = x_pred_train, xout = x_pred_test,
                optns = list(type = "measure", manifold = list(method = "isomap", k = 30), r = 2,
                             layer = 3, hidden = 32, dropout = 0.3, lr = 0.0005,
                             num_epochs = 2000, seed = i))
  err_dfr = sapply(1:nrow(quan_test), function(j) mean((res_dfr$yPred[j, ] - quan_test[j, ])^2))
  
  ###################################### GFR ###################################
  res_gn = grem(y = y, x = x_pred_train, xOut = x_pred_test)
  err_gn = sapply(1:nrow(quan_test), function(j) mean((res_gn$qp[j,] - quan_test[j,])^2))
  
  ##################################### SDR ####################################
  X_SDR = x_pred_train
  Xout_SDR = x_pred_test
  dist.den = as.matrix(dist(quan_train, upper = TRUE, diag = TRUE)) / sqrt(ncol(quan_train))
  complexity = 15
  ygram = gram_matrix(quan_train, complexity = complexity, type = "distribution", kernel = "Gaussian")
  f = get("fopg")
  bhat = f(x = X_SDR, y = ygram, d = 2)$beta
  csd_train = as.matrix(X_SDR) %*% bhat
  csd_test = as.matrix(Xout_SDR) %*% bhat
  rg_SDR = apply(csd_train, 2, function(xxx) diff(range(xxx)))
  res_SDR = lrem(y = y, x = csd_train, xOut = csd_test, optns = list(bwReg = rg_SDR * 0.1))
  err_sdr = sapply(1:nrow(quan_test), function(j) mean((res_SDR$qp[j, ] - quan_test[j, ])^2))

  ############################### Single Index Model ###########################
  res_sid = SIdxDenReg(xin = as.matrix(x_pred_train), qin = quan_train)
  X_SID_train = as.matrix(x_pred_train) %*% res_sid$est; X_SID_test = as.matrix(x_pred_test) %*% res_sid$est;
  res_SID = lrem(y = y, x = X_SID_train, xOut = X_SID_test, optns = list(bwReg = res_sid$bw, lower = 0, upper = 100))
  err_IFR = sapply(1:nrow(quan_test), function(j) mean((res_SID$qp[j,] - quan_test[j,])^2))
  
  ##############################################################################
  ############################## Save Results ##################################
  ##############################################################################
  res = data.frame(q = q,
                   E2M = err_E2M,
                   DFR = err_dfr,
                   GFR = err_gn,
                   SDR = err_sdr,
                   IFR = err_IFR)
  
  return(res)
}

# Stop the parallel cluster
stopCluster(cl)

# Table 4: Mortality
do.call(rbind, result)  %>%
  dplyr::select(-q) %>%
  gather(key = "method", value = Error) %>%
  group_by(method) %>%
  dplyr::summarise(MSPE = sprintf("%.3f", mean(Error)),
                   sd = sprintf("(%.3f)", sd(Error))) %>%
  gather(key = "measure", value = "value", MSPE:sd) %>%
  mutate(method = factor(method, levels = c("E2M", "DFR", "GFR", "SDR", "IFR"))) %>%
  dplyr::select(method, measure, value) %>%
  spread(key = method, value = value)

