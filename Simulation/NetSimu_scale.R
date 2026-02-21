# Set the working directory to the current script location
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Library packages
library(dplyr)
library(ggplot2)
library(tidyverse)

# Load functions for E2M model
source("../code/E2M/E2M.R")
source("../code/E2M/lcm.R")
reticulate::source_python("../code/E2M/E2M.py")

# Load functions for GFR models
function_path = "../code/Network-Regression-with-Graph-Laplacians/src/"
function_sources = list.files(function_path, pattern="*.R$", full.names=TRUE, 
                              ignore.case=TRUE)
sapply(function_sources, source, .GlobalEnv)
source("../code/Network-Regression-with-Graph-Laplacians/severn/functions_needed.R")

# Set up parallel processing
library(doRNG)
library(doParallel)
ncores = detectCores() - 1
cl = makeCluster(ncores) 
registerDoParallel(cl)

# Define parameter grid for simulations
param_grid = expand.grid(n = c(5000, 10000), rrr = 1:200)

# Perform simulations using foreach loop
result = foreach(params = t(param_grid), .packages = c('tidyverse', 'dplyr', 'frechet', 'foreach', 'reshape2', 'vegan', 'reticulate')) %dorng% {
  n = params[1]  # Sample size
  rrr = params[2]  # Replication index
  
  # Load functions for E2M model
  source("../code/E2M/E2M.R")
  source("../code/E2M/lcm.R")
  reticulate::source_python("../code/E2M/E2M.py")
  
  function_path = "../code/Network-Regression-with-Graph-Laplacians/src/"
  function_sources = list.files(function_path, pattern="*.R$", full.names=TRUE, 
                                ignore.case=TRUE)
  sapply(function_sources, source, .GlobalEnv)
  source("../code/Network-Regression-with-Graph-Laplacians/severn/functions_needed.R")
  
  nOut = 200 # Number of test samples
  m = c(5, 5)
  d = c(m[1]*(m[1] - 1)/2, m[1]*m[2], m[2]*(m[2] - 1)/2)
  theta = c(0.5, 0.2, 0.5)
  
  set.seed(rrr) # Set seed for reproducibility
  
  # Generate predictor data for training and test sets
  X = data.frame(
    X1 = runif(n,0,1), X2 = runif(n,-1/2,1/2), X3 = runif(n,1,2),
    X4 = rnorm(n,0,1), X5 = rnorm(n,0,1), X6 = rnorm(n,5,5),
    X7 = sample(c(0,1),n,TRUE,c(0.4,0.6)),
    X8 = sample(c(0,1),n,TRUE,c(0.3,0.7)),
    X9 = sample(c(0,1),n,TRUE,c(0.6,0.3))
  )
  xOut = data.frame(
    X1 = runif(nOut,0,1), X2 = runif(nOut,-1/2,1/2), X3 = runif(nOut,1,2),
    X4 = rnorm(nOut,0,1), X5 = rnorm(nOut,0,1), X6 = rnorm(nOut,5,5),
    X7 = sample(c(0,1),nOut,TRUE,c(0.4,0.6)),
    X8 = sample(c(0,1),nOut,TRUE,c(0.3,0.7)),
    X9 = sample(c(0,1),nOut,TRUE,c(0.6,0.3))
  )
  
  # Generate response data
  L = list()
  LMean = list()
  true_ab = NULL
  for(i in 1:n){
    a1 = 2*sin(pi*X[i,1])*X[i,8] + cos(pi*X[i,2])*(1-X[i,8])
    b1 = 2*X[i,4]^2*X[i,7]+X[i,5]^2*(1-X[i,7])
    
    a2 = sin(pi*X[i,1])*X[i,8] + 2*cos(pi*X[i,2])*(1-X[i,8])
    b2 = X[i,4]^2*X[i,7]+2*X[i,5]^2*(1-X[i,7])
    # true_ab = rbind(true_ab, data.frame(a = a, b = b))
    
    bk1Vec = -rbinom(d[1], 1, theta[1])*rbeta(d[1], shape1 = a1, shape2 = b1)
    bk2Vec = -rbinom(d[2], 1, theta[2])*rbeta(d[2], shape1 = a1, shape2 = b2)
    bk3Vec = -rbinom(d[3], 1, theta[3])*rbeta(d[3], shape1 = a2, shape2 = b2)
    
    temp1 = matrix(0, nrow = m[1], ncol = m[1])
    temp1[lower.tri(temp1)] = bk1Vec
    temp1 = temp1 + t(temp1)
    temp2 = matrix(bk2Vec, nrow = m[1])
    temp3 = matrix(0, nrow = m[2], ncol = m[2])
    temp3[lower.tri(temp3)] = bk3Vec
    temp3 = temp3 + t(temp3)
    temp = rbind(cbind(temp1, temp2), cbind(t(temp2), temp3))
    diag(temp) = -colSums(temp)
    L[[i]] = temp
    
    temp1 = matrix(0, nrow = m[1], ncol = m[1])
    temp1[lower.tri(temp1)] = -rep(theta[1]*a1/(a1+b1), d[1])
    temp1 = temp1 + t(temp1)
    temp2 = matrix(-rep(theta[2]*a1/(a1+b2), d[2]), nrow = m[1])
    temp3 = matrix(0, nrow = m[2], ncol = m[2])
    temp3[lower.tri(temp3)] = -rep(theta[3]*a2/(a2+b2), d[3])
    temp3 = temp3 + t(temp3)
    temp = rbind(cbind(temp1, temp2), cbind(t(temp2), temp3))
    diag(temp) = -colSums(temp)
    LMean[[i]] = temp
  }
  
  # Generate true quantile values for test data
  LMeanOut = list()
  for(i in 1:nOut){
    a1 = 2*sin(pi*xOut[i,1])*xOut[i,8] + cos(pi*xOut[i,2])*(1-xOut[i,8])
    b1 = 2*xOut[i,4]^2*xOut[i,7]+xOut[i,5]^2*(1-xOut[i,7])
    
    a2 = sin(pi*xOut[i,1])*xOut[i,8] + 2*cos(pi*xOut[i,2])*(1-xOut[i,8])
    b2 = xOut[i,4]^2*xOut[i,7]+2*xOut[i,5]^2*(1-xOut[i,7])
    
    temp1 = matrix(0, nrow = m[1], ncol = m[1])
    temp1[lower.tri(temp1)] = -rep(theta[1]*a1/(a1+b1), d[1])
    temp1 = temp1 + t(temp1)
    temp2 = matrix(-rep(theta[2]*a1/(a1+b2), d[2]), nrow = m[1])
    temp3 = matrix(0, nrow = m[2], ncol = m[2])
    temp3[lower.tri(temp3)] = -rep(theta[3]*a2/(a2+b2), d[3])
    temp3 = temp3 + t(temp3)
    temp = rbind(cbind(temp1, temp2), cbind(t(temp2), temp3))
    diag(temp) = -colSums(temp)
    LMeanOut[[i]] = temp
  }
  
  ################################### E2M ####################################
  layer = c(2,3,4,5,6)
  hidden = c(8,16,32,64,128)
  lamb = c(-1e-1,-1e-2,0,1e-2,1e-1)
  res_E2M = E2M(y = L, x = X, xout = xOut,
                optns = list(type = "network", metric = "frobenius",
                             layer = layer, hidden = hidden, 
                             lamb = lamb, lr = 0.0005, dropout = 0.3, 
                             batch_size = 128, axis = -2, num_epochs = 2000, 
                             seed = rrr, n_anchor = 1000L))
  err_E2M = sapply(1:length(LMeanOut), function(j) sum((res_E2M$yPred[[j]] - LMeanOut[[j]])^2))
  
  ##################################### GFR ####################################
  res_gfr = gnr(gl = L, x = X, xOut = xOut)
  err_gn = sapply(1:length(LMeanOut), function(j) sum((res_gfr$predict[[j]] - LMeanOut[[j]])^2))
  
  ##############################################################################
  ########################### Calculate prediction errors ######################
  ##############################################################################
  err_q = data.frame(n = n, r = rrr,
                     E2M = mean(err_E2M),
                     GFR = mean(err_gn)
  )
  
  return(err_q)
}

# Stop the parallel cluster
stopCluster(cl)

# Table 2: Network outputs
do.call(rbind, result)  %>%
  dplyr::select(-r) %>%
  gather(key = "method", value = Error, -c("n")) %>%
  group_by(n, method) %>%
  dplyr::summarise(MSPE = sprintf("%.3f", mean(Error)),
                   sd = sprintf("(%.3f)", sd(Error))) %>%
  gather(key = "measure", value = "value", MSPE:sd) %>%
  mutate(method = factor(method, levels = c("E2M", "GFR"))) %>%
  dplyr::select(n, method, measure, value) %>%
  spread(key = method, value = value)
