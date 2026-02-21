# Set the working directory to the current script location
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Library packages
library(dplyr)
library(tidyverse)

# Load functions for E2M model
source("../../code/E2M/E2M.R")
source("../../code/E2M/lcm.R")
reticulate::source_python('../../code/E2M/E2M.py')

# Load functions for GFR models
function_path = "../../code/Network-Regression-with-Graph-Laplacians/src/"
function_sources = list.files(function_path, pattern="*.R$", full.names=TRUE, 
                              ignore.case=TRUE)
sapply(function_sources, source, .GlobalEnv)
source("../../code/Network-Regression-with-Graph-Laplacians/severn/functions_needed.R")

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
source("../../code/Single-Index-Frechet/SIdxNetReg.R")

# Load Taxi Data
taxi = readRDS("taxi_data.RData")
y_nw = taxi$taxigl
x_pred = taxi$x_pred
meanLapM_y = sum((Reduce("+", y_nw) / length(y_nw))^2)

# Set up parallel processing
library(doRNG)
library(doParallel)
ncores = detectCores() - 1
cl = makeCluster(ncores) 
length(cl)
registerDoParallel(cl)

# Define parameter grid for simulations
param_grid = expand.grid(q = 1:100, k = 1:10)

# Perform simulations using foreach loop
result = foreach(params = t(param_grid), .packages = c('tidyverse', 'dplyr', 'frechet', 'foreach', 'reshape2', 'vegan', 'reticulate')) %dorng% {
  # Reload necessary functions for each iteration
  source("../../code/E2M/E2M.R")
  source("../../code/E2M/lcm.R")
  reticulate::source_python('../../code/E2M/E2M.py')
  function_path = "../../code/Network-Regression-with-Graph-Laplacians/src/"
  function_sources = list.files(function_path, pattern="*.R$", full.names=TRUE, 
                                ignore.case=TRUE)
  sapply(function_sources, source, .GlobalEnv)
  source("../../code/Network-Regression-with-Graph-Laplacians/severn/functions_needed.R")
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
  source("../../code/Single-Index-Frechet/SIdxNetReg.R")  
  
  q = params[1] # Replication index
  k = params[2] # k-th fold
  n = nrow(x_pred) # Sample size
  
  # Split data into training and test sets
  cv_fold = data.frame(rd_ind = sample(1:n,n,replace = FALSE),
                       fold = rep(1:10,length.out=n))
  
  ind_test = cv_fold[cv_fold$fold==k,1]
  ind_remain = cv_fold[cv_fold$fold!=k,1]
  
  y_nw_train = y_nw[ind_remain]
  y_nw_test = y_nw[ind_test]
  x_pred_train = x_pred[ind_remain,]
  x_pred_test = x_pred[ind_test,]
  
  x_pred_q = x_pred
  x_pred_train_mean = as.vector(colMeans(x_pred_train))
  x_pred_train_sd = as.vector(apply(x_pred_train, 2, sd))
  
  x_pred_train_mean[which(names(x_pred_train) %in% c("ind_year", "MTWT", "FS"))] = 0
  x_pred_train_sd[which(names(x_pred_train) %in% c("ind_year", "MTWT", "FS"))] = 1
  
  # standardize individually
  x_pred_train = t((t(x_pred_train) - x_pred_train_mean)/x_pred_train_sd)
  x_pred_test = t((t(x_pred_test) - x_pred_train_mean)/x_pred_train_sd)
  
  ################################### E2M ######################################
  layer = c(2,3,4,5,6)
  hidden = c(8,16,32,64,128)
  lamb = c(-1e-1,-1e-2,0,1e-2,1e-1)
  res_E2M = E2M(y = y_nw_train, x = x_pred_train, xout = x_pred_test,
                optns = list(type = "network", metric = "frobenius", layer = layer,
                             hidden = hidden, lr = 0.0005, num_epochs = 2000,
                             dropout = 0.3, norm_axis = -2, seed = q))
  err_E2M = sapply(1:length(y_nw_test), function(j) sum((res_E2M$yPred[[j]] - y_nw_test[[j]])^2))/meanLapM_y
  
  ##################################### DFR ####################################
  res_dfr = DFR(y = y_nw_train, x = x_pred_train, xout = x_pred_test, 
                optns = list(type = "laplacian", manifold = list(method = "isomap", k = 20),
                             r = 2, layer = 4, hidden = 64, dropout = 0.3, 
                             lr = 0.001, num_epochs = 2000, seed = q))
  err_dfr = sapply(1:length(y_nw_test), function(j) sum((res_dfr$yPred[[j]] - y_nw_test[[j]])^2))/meanLapM_y
  
  ###################################### GFR ###################################
  res_gn = gnr(gl = y_nw_train, x = x_pred_train, xOut = x_pred_test)
  err_gn = sapply(1:length(y_nw_test), function(j) sum((res_gn$predict[[j]] - y_nw_test[[j]])^2))/meanLapM_y
  
  ##################################### SDR ####################################
  X_SDR = x_pred_train
  Xout_SDR = x_pred_test
  y_mat = do.call(rbind, lapply(y_nw_train, function(i) as.numeric(i)))
  dist.den = as.matrix(dist(y_mat, upper = T, diag = T))
  complexity = 1/2/sum((dist.den[upper.tri(dist.den)])^2/meanLapM_y)*choose(n,2)
  ygram = gram_matrix(y_nw_train, complexity = complexity, type="spd", kernel="Gaussian")
  f = get("fopg")
  bhat = f(x=X_SDR, y=ygram, d=2)$beta
  csd_train = as.matrix(X_SDR)%*%bhat; csd_test = as.matrix(Xout_SDR)%*%bhat;
  rg_SDR = apply(csd_train, 2, function(xxx) range(xxx)[2]-range(xxx)[1])
  res_SDR = lnr(gl = y_nw_train, x = csd_train, xOut = csd_test,
                optns = list(bwReg = rg_SDR*0.1))
  err_sdr = sapply(1:length(y_nw_test),
                   function(j) {
                     sum((res_SDR$predict[[j]] - y_nw_test[[j]])^2)
                   })/meanLapM_y
  
  ############################### Single Index Model ###########################
  m = dim(y_nw_train[[1]])[1]
  y_mat = array(0, c(m, m, length(y_nw_train)))
  for(j in 1:length(y_nw_train)){
    y_mat[,,j] = y_nw_train[[j]]
  }
  res_sid = SIdxNetReg(xin = as.matrix(x_pred_train), Min = y_mat, bw = 1.0937, M = 4, iter = 100)
  X_SID_train = as.matrix(x_pred_train) %*% res_sid$est; X_SID_test = as.matrix(x_pred_test) %*% res_sid$est;
  rg_SID = apply(X_SID_train, 2, function(xxx) range(xxx)[2]-range(xxx)[1])
  res_IFR = lnr(gl = y_nw_train, x = X_SID_train, xOut = X_SID_test, optns = list(bwReg = rg_SID*0.1))
  
  err_IFR = sapply(1:nrow(x_pred_test), function(j){
    sum((y_nw_test[[j]] - res_IFR$predict[[j]])^2)
  })/meanLapM_y
  
  ##############################################################################
  ############################## Save Results ##################################
  ##############################################################################
  err_q = data.frame(q = q, k = k, 
                     E2M = mean(err_E2M),
                     DFR = mean(err_dfr),
                     GFR = mean(err_gn),
                     SDR = mean(err_sdr),
                     IFR = mean(err_IFR)
  )
  
  return(err_q)
}

# Stop the parallel cluster
stopCluster(cl)

# Table 4: Taxi
do.call(rbind, result)  %>%
  gather(key = "method", value = Error, -c("q", "k")) %>%
  group_by(method,q) %>%
  dplyr::summarise(Error = mean(Error)*1000) %>%
  group_by(method) %>%
  dplyr::summarise(MSPE = sprintf("%.3f", mean(Error)),
                   sd = sprintf("(%.3f)", sd(Error))) %>%
  gather(key = "measure", value = "value", MSPE:sd) %>%
  mutate(method = factor(method, levels = c("E2M", "DFR", "GFR", "SDR", "IFR"))) %>%
  dplyr::select(method, measure, value) %>%
  spread(key = method, value = value)
