# Set the working directory to the current script location
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Library packages
library(dplyr)
library(tidyverse)
library(KScorrect)
library(EnvStats)
library(mvtnorm)

# Load functions for E2M model
source("../code/E2M/E2M.R")
source("../code/E2M/lcm.R")
reticulate::source_python("../code/E2M/E2M.py")

# Load functions for GFR model
function_path = "../code/Wasserstein-regression-with-empirical-measures/code/"
function_sources = list.files(function_path, pattern="*.R$", full.names=TRUE, 
                              ignore.case=TRUE)
sapply(function_sources, source, .GlobalEnv)

# Load functions for SDR model
function_path = "../code/DR4FrechetReg/Functions"
function_sources = list.files(function_path, pattern="*.R$", full.names=TRUE, 
                              ignore.case=TRUE)
sapply(function_sources, source, .GlobalEnv)

# Load functions for DFR model and required Python scripts
function_path = "../code/DFR"
function_sources = list.files(function_path, pattern="*.R$", full.names=TRUE, 
                              ignore.case=TRUE)
sapply(function_sources, source, .GlobalEnv)
reticulate::source_python('../code/DFR/NN_class_new_batchnorm.py')
reticulate::source_python('../code/DFR/DNN_dy_layers.py')

# Load functions for IFR model
source("../code/Single-Index-Frechet/SIdxDenReg.R")

# Set up parallel processing
library(doRNG)
library(doParallel)
ncores = detectCores() - 1
cl = makeCluster(ncores) 
length(cl)
registerDoParallel(cl)

# Define parameter grid for simulations
param_grid = expand.grid(n = c(50,100,200,500,1000,2000), rrr = 1:100)

# Perform simulations using foreach loop
result = foreach(params = t(param_grid), .packages = c('tidyverse', 'dplyr', 'frechet', 'foreach', 'reshape2', 'vegan', 'reticulate', 'EnvStats', 'mvtnorm')) %dorng% {
  n = params[1]  # Sample size
  rrr = params[2]  # Replication index
  
  # Reload necessary functions for each iteration
  source("../code/E2M/E2M.R")
  reticulate::source_python('../code/E2M/E2M.py')
  function_path = "../code/Wasserstein-regression-with-empirical-measures/code/"
  function_sources = list.files(function_path, pattern="*.R$", full.names=TRUE, 
                                ignore.case=TRUE)
  sapply(function_sources, source, .GlobalEnv)
  function_path = "../code/DR4FrechetReg/Functions"
  function_sources = list.files(function_path, pattern="*.R$", full.names=TRUE, 
                                ignore.case=TRUE)
  sapply(function_sources, source, .GlobalEnv)
  function_path = "../code/DFR"
  function_sources = list.files(function_path, pattern="*.R$", full.names=TRUE, 
                                ignore.case=TRUE)
  sapply(function_sources, source, .GlobalEnv)
  reticulate::source_python('../code/DFR/NN_class_new_batchnorm.py')
  reticulate::source_python('../code/DFR/DNN_dy_layers.py')
  source("../code/Single-Index-Frechet/SIdxDenReg.R")
  
  # Define simulation parameters
  N = 100  # Number of observations per sample
  nOut = 200  # Number of test samples
  sigma0 = 3  # Base standard deviation
  kappa = 1  # Shape parameter for gamma distribution
  
  set.seed(rrr)  # Set seed for reproducibility
  mu = rep(0,12)   # Mean vector for each variable
  # Define a positive-definite symmetric covariance matrix (Sigma)
  Sigma = toeplitz(0.8^(0:11))
  X = rmvnorm(n = n, mean = mu, sigma = Sigma)
  xOut = rmvnorm(n = nOut, mean = mu, sigma = Sigma)
  
  # Generate response data
  y = list()
  yMean = matrix(nrow = n, ncol = N-1)
  
  for(i in 1:n) {
    # Generate expected mean and standard deviation
    expect_eta_Z1 = sum(X[i,] * c(1,0,1,0,1,-1,1,-1,1,0,1,0))
    expect_sigma_Z1 = 2
    # Sample mean and standard deviation
    mu1 = rnorm(1, mean = expect_eta_Z1, sd = 0.5)
    sigma1 = rgamma(1, shape = expect_sigma_Z1^2/kappa, scale = kappa/expect_sigma_Z1)
    
    # Generate response and true quantile values
    y[[i]] = sort(rnorm(N, mean = mu1, sd = sigma1))
    yMean[i, ] = qnorm(c(1:(N-1))/N,
                       mean = expect_eta_Z1, 
                       sd = expect_sigma_Z1)
  }
  
  # Generate true quantile values for test data
  yMeanOut = matrix(nrow=nOut, ncol=N-1)
  for (i in 1:nOut) {
    expect_eta_Z1 = sum(xOut[i,] * c(1,0,1,0,1,-1,1,-1,1,0,1,0))
    expect_sigma_Z1 = 2
    
    yMeanOut[i, ] = qnorm(c(1:(N-1))/N, mean = expect_eta_Z1, sd = expect_sigma_Z1)
  }

  ################################### E2M ######################################
  layer = c(2,3,4,5,6)
  hidden = c(8,16,32,64,128)
  lamb = c(-1e-1,-1e-2,0,1e-2,1e-1)
  res_E2M = E2M(y = y, x = X, xout = xOut, 
                    optns = list(type = "measure", metric = "wasserstein",
                                 layer = layer, hidden = hidden, 
                                 lamb = lamb, lr = 0.0005, dropout = 0.3, 
                                 axis = -1, num_epochs = 2000, seed = rrr))
  
  ##################################### GFR ####################################
  res_gfr = grem(y = y, x = X, xOut = xOut)
  
  ##################################### DFR ####################################
  res_dfr = DFR(y = y, x = X, xout = xOut, 
                optns = list(type = "measure", 
                             manifold = list(method = "isomap", k = 0.1 * n), 
                             r = 2, layer = 4, hidden = 32, 
                             dropout = 0.3, lr = 0.0005, 
                             num_epochs = 2000, seed = rrr))
  
  ##################################### SDR ####################################
  y_mat = do.call(rbind, y)
  dist.den = as.matrix(dist(y_mat, upper = T, diag = T))/sqrt(N)
  ## standardize individually
  X_SDR = apply(X,2,function(x) return((x-mean(x))/sd(x))) 
  Xout_SDR = apply(xOut,2,function(x) return((x-mean(x))/sd(x))) 
  
  y_mat = do.call(rbind, y)
  complexity = 1/2/(sum(dist.den[upper.tri(dist.den)]^2)/choose(n,2))
  ygram = gram_matrix(y_mat, complexity = complexity, 
                      type="distribution", kernel="Gaussian")
  f = get("fopg")
  bhat = f(x=X_SDR, y=ygram, d=2)$beta
  csd_train = as.matrix(X_SDR)%*%bhat; csd_test = as.matrix(Xout_SDR)%*%bhat;
  
  rg_SDR = apply(csd_train, 2, function(xxx) range(xxx)[2]-range(xxx)[1])
  res_SDR = lrem(y = y, x = csd_train, xOut = csd_test, optns = list(bwReg = rg_SDR*0.1))
  
  ############################### Single Index Model ###########################
  y_mat = do.call(rbind, y)
  q_mat = t(apply(y_mat, 1, sort))
  res_ifr = SIdxDenReg(xin = as.matrix(X), qin = q_mat)
  X_IFR_train = as.matrix(X) %*% res_ifr$est; X_IFR_test = as.matrix(xOut) %*% res_ifr$est
  res_IFR = lrem(y = y, x = X_IFR_train, xOut = X_IFR_test, optns = list(bwReg = res_ifr$bw))
  
  ##############################################################################
  ########################### Calculate prediction errors ######################
  ##############################################################################
  err_q = data.frame(
    n = n, r = rrr,
    E2M = sum((res_E2M$yPred[, -N] - yMeanOut)^2) / ((N - 1) * nOut),
    DFR = sum((res_dfr$yPred[, -N] - yMeanOut)^2) / ((N - 1) * nOut),
    GFR = sum((res_gfr$qp[, -N] - yMeanOut)^2) / ((N - 1) * nOut),
    SDR = sum((res_SDR$qp[, -N] - yMeanOut)^2) / ((N - 1) * nOut),
    IFR = sum((res_IFR$qp[, -N] - yMeanOut)^2) / ((N - 1) * nOut)
  )

  return(err_q)
}

# Stop the parallel cluster
stopCluster(cl)

# Table 9: Distributional outputs
do.call(rbind, result)  %>%
  dplyr::select(-r) %>%
  gather(key = "method", value = Error, -c("n")) %>%
  group_by(n, method) %>%
  dplyr::summarise(MSPE = sprintf("%.3f", mean(Error)),
                   sd = sprintf("(%.3f)", sd(Error))) %>%
  gather(key = "measure", value = "value", MSPE:sd) %>%
  mutate(method = factor(method, levels = c("E2M", "DFR", "GFR", "SDR", "IFR"))) %>%
  dplyr::select(n, method, measure, value) %>%
  spread(key = method, value = value)
