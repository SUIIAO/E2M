# Set the working directory to the current script location
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Library packages
library(dplyr)
library(tidyverse)
library(KScorrect)
library(EnvStats)

# Load functions for E2M model
source("../code/E2M/E2M.R")
source("../code/E2M/lcm.R")
reticulate::source_python("../code/E2M/E2M.py")

# Load functions for GFR model
function_path = "../code/Wasserstein-regression-with-empirical-measures/code/"
function_sources = list.files(function_path, pattern="*.R$", full.names=TRUE, 
                              ignore.case=TRUE)
sapply(function_sources, source, .GlobalEnv)

# Set up parallel processing
library(doRNG)
library(doParallel)
ncores = detectCores() - 1
cl = makeCluster(ncores) 
length(cl)
registerDoParallel(cl)

# Define parameter grid for simulations
param_grid = expand.grid(n = c(10000), rrr = 1:200)

# Perform simulations using foreach loop
result = foreach(params = t(param_grid), .packages = c('tidyverse', 'dplyr', 'frechet', 'foreach', 'reshape2', 'vegan', 'reticulate', 'EnvStats')) %dorng% {
  n = params[1]  # Sample size
  rrr = params[2]  # Replication index
  
  # Load functions for E2M model
  source("../code/E2M/E2M.R")
  source("../code/E2M/lcm.R")
  reticulate::source_python('../code/E2M/E2M.py')
  function_path = "../code/Wasserstein-regression-with-empirical-measures/code/"
  function_sources = list.files(function_path, pattern="*.R$", full.names=TRUE, 
                                ignore.case=TRUE)
  sapply(function_sources, source, .GlobalEnv)

  # Define simulation parameters
  N = 100  # Number of observations per sample
  nOut = 200  # Number of test samples
  sigma0 = 3  # Base standard deviation
  kappa = 1  # Shape parameter for gamma distribution
  
  set.seed(rrr)  # Set seed for reproducibility
  
  # Generate predictor data for training and test sets
  X = data.frame(X1 = runif(n, -1, 0), X2 = runif(n, -1, 0), 
                 X3 = runif(n, 0, 1), X4 = runif(n, 0, 1),
                 
                 X5 = rgamma(n, 2, 2), X6 = rgamma(n, 3, 2), 
                 X7 = rgamma(n, 4, 2), X8 = rgamma(n, 5, 2),
                 
                 X9 = sample(c(0,1),n,TRUE,c(0.4,0.6)), 
                 X10 = sample(c(0,1),n,TRUE,c(0.5,0.5)), 
                 X11 = sample(c(0,1),n,TRUE,c(0.6,0.4)),
                 X12 = sample(c(0,1),n,TRUE,c(0.7,0.3))
  )
  xOut = data.frame(X1 = runif(nOut, -1, 0), X2 = runif(nOut, -1, 0), 
                    X3 = runif(nOut, 0, 1), X4 = runif(nOut, 0, 1),
                    
                    X5 = rgamma(nOut, 2, 2), X6 = rgamma(nOut, 3, 2), 
                    X7 = rgamma(nOut, 4, 2), X8 = rgamma(nOut, 5, 2),
                    
                    X9 = sample(c(0,1),nOut,TRUE,c(0.4,0.6)), 
                    X10 = sample(c(0,1),nOut,TRUE,c(0.5,0.5)), 
                    X11 = sample(c(0,1),nOut,TRUE,c(0.6,0.4)),
                    X12 = sample(c(0,1),nOut,TRUE,c(0.7,0.3))
  )
  
  # Generate response data
  y = list()
  yMean = matrix(nrow = n, ncol = N-1)
  
  for(i in 1:n) {
    # Generate expected mean and standard deviation
    expect_eta_Z1 = 2 + 2 * cos(pi*X[i,1])^2 + sin(pi*X[i,2])^2*X[i,9] + sqrt(X[i,5])* sqrt(X[i,6]) * (1-X[i,9])
    expect_sigma_Z1 = 1 + cos(pi/2*X[i,2]) + sin(pi*X[i,3]) * X[i,10] + sqrt(X[i,6])* sqrt(X[i,7])/3 * (1-X[i,10])
    
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
    expect_eta_Z1 = 2 + 2 * cos(pi*xOut[i,1])^2 + sin(pi*xOut[i,2])^2*xOut[i,9] + sqrt(xOut[i,5])*sqrt(xOut[i,6]) * (1-xOut[i,9])
    expect_sigma_Z1 = 1 + cos(pi/2*xOut[i,2]) + sin(pi*xOut[i,3]) * xOut[i,10] + sqrt(xOut[i,6])*sqrt(xOut[i,7])/3 * (1-xOut[i,10])
    
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
                             batch_size = 128, axis = -2, num_epochs = 2000, 
                             seed = rrr, n_anchor = 1000L))
  
  ##################################### GFR ####################################
  res_gfr = grem(y = y, x = X, xOut = xOut)
  
  ##############################################################################
  ########################### Calculate prediction errors ######################
  ##############################################################################
  err_q = data.frame(
    n = n, r = rrr,
    E2M = sum((res_E2M$yPred[, -N] - yMeanOut)^2) / ((N - 1) * nOut),
    GFR = sum((res_gfr$qp[, -N] - yMeanOut)^2) / ((N - 1) * nOut))

  return(err_q)
}

# Stop the parallel cluster
stopCluster(cl)

# Table 2: Distributional outputs
do.call(rbind, result)  %>%
  dplyr::select(-r) %>%
  gather(key = "method", value = Error, -c("n")) %>%
  group_by(n, method) %>%
  dplyr::summarise(MSPE = sprintf("%.4f", mean(Error)),
                   sd = sprintf("(%.4f)", sd(Error))) %>%
  gather(key = "measure", value = "value", MSPE:sd) %>%
  mutate(method = factor(method, levels = c("E2M", "GFR"))) %>%
  dplyr::select(n, method, measure, value) %>%
  spread(key = method, value = value)
