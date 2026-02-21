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

# Frechet Forest
source("../code/Code_RFWLFR/FRFPackage2.R")
source("../code/Code_RFWLFR/main.R")

# Set up parallel processing
library(doRNG)
library(doParallel)
ncores = detectCores() - 1
cl = makeCluster(ncores) 
length(cl)
registerDoParallel(cl)

# Define parameter grid for simulations
param_grid = expand.grid(n = c(50,100,200,500,1000,2000), rrr = 1:500)

# Perform simulations using foreach loop
result = foreach(params = t(param_grid), .packages = c('tidyverse', 'dplyr', 'frechet', 'foreach', 'reshape2', 'vegan', 'reticulate', 'EnvStats')) %dorng% {
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
  source("../code/Code_RFWLFR/FRFPackage2.R")
  source("../code/Code_RFWLFR/main.R")
  
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
 
  ################################# Random Forest ##############################
  X_forest = list()
  X_forest$type = "scalar"
  X_forest$id = 1:n
  X_forest$X = as.matrix(X)
  n_estimators = 100
  y_mat = do.call(rbind, y)
  q_mat = t(apply(y_mat, 1, sort))
  Y_forest = list()
  Y_forest$type = "distribution"
  Y_forest$id = 1:n
  Y_forest$Y = q_mat
  X_new_forest = list()
  X_new_forest$type = "scalar"
  X_new_forest$id = 1:nOut
  X_new_forest$X = as.matrix(xOut)
  p = ncol(X)
  nqSup = ncol(q_mat) 
  qSup = seq(0,1,length.out = nqSup)
  
  # RFWLLFR
  res_rfwlfr = rfwllfr(r = rrr, Scalar=X_forest, Y=Y_forest, Xout=X_new_forest, mtry=ceiling(4/5*p), deep=5, ntree=n_estimators, ncores=1)
  
  ##############################################################################
  ########################### Calculate prediction errors ######################
  ##############################################################################
  err_q = data.frame(
    n = n, r = rrr,
    E2M = sum((res_E2M$yPred[, -N] - yMeanOut)^2) / ((N - 1) * nOut),
    DFR = sum((res_dfr$yPred[, -N] - yMeanOut)^2) / ((N - 1) * nOut),
    GFR = sum((res_gfr$qp[, -N] - yMeanOut)^2) / ((N - 1) * nOut),
    SDR = sum((res_SDR$qp[, -N] - yMeanOut)^2) / ((N - 1) * nOut),
    IFR = sum((res_IFR$qp[, -N] - yMeanOut)^2) / ((N - 1) * nOut),
    RFWLFR = sum((res_rfwlfr$res[, -N] - yMeanOut)^2) / ((N - 1) * nOut)
  )

  return(err_q)
}

# Stop the parallel cluster
stopCluster(cl)

# Table 1: Distributional outputs
do.call(rbind, result)  %>%
  dplyr::select(-r) %>%
  gather(key = "method", value = Error, -c("n")) %>%
  group_by(n, method) %>%
  dplyr::summarise(MSPE = sprintf("%.3f", mean(Error)),
                   sd = sprintf("(%.3f)", sd(Error))) %>%
  gather(key = "measure", value = "value", MSPE:sd) %>%
  mutate(method = factor(method, levels = c("E2M", "DFR", "GFR", "SDR", 
                                            "IFR", "RFWLFR"))) %>%
  dplyr::select(n, method, measure, value) %>%
  spread(key = method, value = value) %>%
  knitr::kable(format = "latex")
