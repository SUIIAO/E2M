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
source("../code/Single-Index-Frechet/SIdxNetReg.R")

# Frechet Forest
source("../code/Code_RFWLFR/shape_revise.R")
source("../code/Code_RFWLFR/FRFPackage2.R")
source("../code/Code_RFWLFR/main.R")

# Set up parallel processing
library(doRNG)
library(doParallel)
ncores = detectCores() - 1
cl = makeCluster(ncores) 
registerDoParallel(cl)

# Define parameter grid for simulations
param_grid = expand.grid(n = c(500,1000,2000), rrr = 1:500)

# Perform simulations using foreach loop
result = foreach(params = t(param_grid), .packages = c('tidyverse', 'dplyr', 'frechet', 'foreach', 'reshape2', 'vegan', 'reticulate')) %dorng% {
  n = params[1]  # Sample size
  rrr = params[2]  # Replication index
  
  # Reload necessary functions for each iteration
  source("../code/E2M/E2M.R")
  reticulate::source_python('../code/E2M/E2M.py')
  function_path = "../code/Network-Regression-with-Graph-Laplacians/src/"
  function_sources = list.files(function_path, pattern="*.R$", full.names=TRUE, 
                                ignore.case=TRUE)
  sapply(function_sources, source, .GlobalEnv)
  source("../code/Network-Regression-with-Graph-Laplacians/severn/functions_needed.R")
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
  source("../code/Single-Index-Frechet/SIdxNetReg.R")
  source("../code/Code_RFWLFR/shape_revise.R")
  source("../code/Code_RFWLFR/FRFPackage2.R")
  source("../code/Code_RFWLFR/main.R")
  
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
  
  ################################### E2M ######################################
  layer = c(2,3,4,5,6)
  hidden = c(8,16,32,64,128)
  lamb = c(-1e-1,-1e-2,0,1e-2,1e-1)
  res_E2M = E2M(y = L, x = X, xout = xOut,
                optns = list(type = "network", metric = "frobenius",
                             layer = layer, hidden = hidden, 
                             lamb = lamb, lr = 0.0005, dropout = 0.3, 
                             axis = -2, num_epochs = 2000, seed = rrr))
  err_E2M = sapply(1:length(LMeanOut), function(j) sum((res_E2M$yPred[[j]] - LMeanOut[[j]])^2))
  
  ##################################### GFR ####################################
  res_gfr = gnr(gl = L, x = X, xOut = xOut)
  err_gn = sapply(1:length(LMeanOut), function(j) sum((res_gfr$predict[[j]] - LMeanOut[[j]])^2))
  
  ##################################### DFR ####################################
  res_dfr = DFR(y = L, x = X, xout = xOut, 
                optns = list(type = "laplacian", manifold = list(method = "isomap", k = 0.1*n), r = 2, 
                             layer = 4, hidden = 32, lr = 0.0005,  
                             num_epochs = 2000, seed = rrr))
  err_dfr = sapply(1:length(LMeanOut), function(j) sum((res_dfr$yPred[[j]] - LMeanOut[[j]])^2))
  
  ##################################### SDR ####################################
  X_SDR = apply(X,2,function(x) return((x-mean(x))/sd(x)))
  Xout_SDR = apply(xOut,2,function(x) return((x-mean(x))/sd(x)))
  
  y_mat = do.call(rbind, lapply(L, function(i) as.numeric(i)))
  dist.den = as.matrix(dist(y_mat, upper = T, diag = T))
  complexity = 1/2/(sum(dist.den[upper.tri(dist.den)])/choose(n,2))
  ygram = gram_matrix(L, complexity = complexity, type="spd", kernel="Gaussian")
  f = get("fopg")
  bhat = f(x=X_SDR, y=ygram, d=2)$beta
  csd_train = as.matrix(X_SDR)%*%bhat; csd_test = as.matrix(Xout_SDR)%*%bhat;
  
  rg_SDR = apply(csd_train, 2, function(xxx) range(xxx)[2]-range(xxx)[1])
  res_SDR = lnr(gl = L, x = csd_train, xOut = csd_test, optns = list(bwReg = rg_SDR*0.1))
  err_SDR = sapply(1:length(LMeanOut), function(j) sum((res_SDR$predict[[j]] - LMeanOut[[j]])^2))
  
  ############################### Single Index Model ###########################
  m = dim(L[[1]])[1]
  y_mat = array(0, c(m, m, length(L)))
  for(j in 1:length(L)){
    y_mat[,,j] = L[[j]]
  }
  res_sid = SIdxNetReg(xin = as.matrix(X), Min = y_mat, bw = 1, M = 4, iter = 100)
  X_SID_train = as.matrix(X) %*% res_sid$est; X_SID_test = as.matrix(xOut) %*% res_sid$est;
  rg_SID = apply(X_SID_train, 2, function(xxx) range(xxx)[2]-range(xxx)[1])
  res_IFR = lnr(gl = L, x = X_SID_train, xOut = X_SID_test, optns = list(bwReg = rg_SID*0.1))
  
  err_IFR = sapply(1:length(LMeanOut), function(j){
    sum((LMeanOut[[j]] - res_IFR$predict[[j]])^2)
  })
  
  ################################# Random Forest ##############################
  X_forest = list()
  X_forest$type = "scalar"
  X_forest$id = 1:n
  X_forest$X = as.matrix(X)
  Y_forest = list()
  Y_forest$type = "laplacian"
  Y_forest$id = 1:n
  m = dim(L[[1]])[1]
  y_mat = array(0, c(m, m, length(L)))
  for(j in 1:length(L)){
    y_mat[,,j] = L[[j]]
  }
  Y_forest$Y = y_mat
  X_new_forest = list()
  X_new_forest$type = "scalar"
  X_new_forest$id = 1:nOut
  X_new_forest$X = as.matrix(xOut)
  p = ncol(X)
  method = "Euclidean"
  
  # RFWLLFR
  res_rfwlfr = rfwllfr(r = rrr, Scalar=X_forest, Y=Y_forest, Xout=X_new_forest, mtry=ceiling(4/5*p), deep=5, ntree=100, ncores=1)
  err_RF = mean(sapply(1:nOut, function(j){
    sum((LMeanOut[[j]] - res_rfwlfr$res[,,j])^2)
  }))
  
  ##############################################################################
  ########################### Calculate prediction errors ######################
  ##############################################################################
  err_q = data.frame(n = n, r = rrr,
                     E2M = mean(err_E2M), DFR = mean(err_dfr),
                     GFR = mean(err_gn), SDR = mean(err_SDR),
                     IFR = mean(err_IFR), RFWLFR = mean(err_RF)
  )
  return(err_q)
}

# Stop the parallel cluster
stopCluster(cl)

# Table 1: Network outputs
do.call(rbind, result)  %>%
  dplyr::select(-r) %>%
  gather(key = "method", value = Error, -c("n")) %>%
  group_by(n, method) %>%
  dplyr::summarise(MSPE = sprintf("%.3f", mean(Error)),
                   sd = sprintf("(%.3f)", sd(Error))) %>%
  gather(key = "measure", value = "value", MSPE:sd) %>%
  mutate(method = factor(method, levels = c("E2M", "DFR", "GFR", "SDR", "IFR", "RFWLFR"))) %>%
  dplyr::select(n, method, measure, value) %>%
  spread(key = method, value = value)
