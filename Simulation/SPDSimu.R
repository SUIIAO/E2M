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

# Load functions for DFR model and required Python scripts
function_path = "../code/DFR"
function_sources = list.files(function_path, pattern="*.R$", full.names=TRUE, 
                              ignore.case=TRUE)
sapply(function_sources, source, .GlobalEnv)
reticulate::source_python('../code/DFR/NN_class_new_batchnorm.py')
reticulate::source_python('../code/DFR/DNN_dy_layers.py')

# Frechet Forest
source("../code/Code_RFWLFR/shape_revise.R")
source("../code/Code_RFWLFR/FRFPackage2.R")
source("../code/Code_RFWLFR/main.R")
method="Power"
alpha = 1/2

# Set up parallel processing
library(doRNG)
library(doParallel)
ncores = detectCores() - 1
cl = makeCluster(ncores) 
registerDoParallel(cl)

# Define parameter grid for simulations
# Define parameter grid for simulations
param_grid = expand.grid(n = c(500,1000,2000), rrr = 1:500)

# Perform simulations using foreach loop
result = foreach(params = t(param_grid), .packages = c('tidyverse', 'dplyr', 'frechet', 'foreach', 'reshape2', 'vegan', 'reticulate')) %dorng% {
  
  n = params[1]  # Sample size
  rrr = params[2]  # Replication index
  
  # Reload necessary functions for each iteration
  # Load functions for E2M model
  source("../code/E2M/E2M.R")
  source("../code/E2M/lcm.R")
  reticulate::source_python('../code/E2M/E2M.py')
  function_path = "../code/Network-Regression-with-Graph-Laplacians/src/"
  function_sources = list.files(function_path, pattern="*.R$", full.names=TRUE, 
                                ignore.case=TRUE)
  sapply(function_sources, source, .GlobalEnv)
  source("../code/Network-Regression-with-Graph-Laplacians/severn/functions_needed.R")
  function_path = "../code/DFR"
  function_sources = list.files(function_path, pattern="*.R$", full.names=TRUE, 
                                ignore.case=TRUE)
  sapply(function_sources, source, .GlobalEnv)
  reticulate::source_python('../code/DFR/NN_class_new_batchnorm.py')
  reticulate::source_python('../code/DFR/DNN_dy_layers.py')
  
  # Frechet Forest
  source("../code/Code_RFWLFR/shape_revise.R")
  source("../code/Code_RFWLFR/FRFPackage2.R")
  source("../code/Code_RFWLFR/main.R")
  
  nOut = 200 # Number of test samples
  m = 5
  
  set.seed(rrr) # Set seed for reproducibility
  
  # Generate predictor data for training and test sets
  X = data.frame(
    X1 = runif(n,0,1), X2 = runif(n,-1/2,1/2), X3 = runif(n,1,2),
    X4 = rgamma(n,3,2), X5 = rgamma(n,4,2), X6 = rgamma(n,5,2),
    X7 = rnorm(n,0,1), X8 = rnorm(n,0,1), X9 = rnorm(n,0,1),
    X10 = sample(c(0,1),n,TRUE,c(0.4,0.6)),
    X11 = sample(c(0,1),n,TRUE,c(0.5,0.5)),
    X12 = sample(c(0,1),n,TRUE,c(0.6,0.4))
  )
  Xout = data.frame(
    X1 = runif(nOut,0,1), X2 = runif(nOut,-1/2,1/2), X3 = runif(nOut,1,2),
    X4 = rgamma(nOut,3,2), X5 = rgamma(nOut,4,2), X6 = rgamma(nOut,5,2),
    X7 = rnorm(nOut,0,1), X8 = rnorm(nOut,0,1), X9 = rnorm(nOut,0,1),
    X10 = sample(c(0,1),nOut,TRUE,c(0.4,0.6)),
    X11 = sample(c(0,1),nOut,TRUE,c(0.5,0.5)),
    X12 = sample(c(0,1),nOut,TRUE,c(0.6,0.4))
  )
  
  # Generate response data
  L = list()
  LMean = list()
  true_ab = NULL
  for(i in 1:n){
    D = c(
      sin(X[i,1]*pi)*X[i,10] + cos(X[i,2]*pi)*(1-X[i,10]), 
      sin(X[i,1]*pi)*cos(X[i,2]*pi),
      (X[i,4]/X[i,5])/10 * X[i,11] + sqrt(X[i,5]/X[i,4])/10 * (1-X[i,11]), 
      sqrt(abs(X[i,7]*X[i,8]))/5,
      sqrt(abs(X[i,9]/X[i,6]))/3
    )
    
    L[[i]] =  rWishart(1, df = m + 1, Sigma = diag(D^2))[,,1]
    LMean[[i]] = diag(D^2) * (m+1)
  }
  
  # Generate true quantile values for test data
  LMeanOut_power = list()
  for(i in 1:nOut){
    D = c(
      sin(Xout[i,1]*pi)*Xout[i,10] + cos(Xout[i,2]*pi)*(1-Xout[i,10]), 
      sin(Xout[i,1]*pi)*cos(Xout[i,2]*pi),
      (Xout[i,4]/Xout[i,5])/10 * Xout[i,11] + sqrt(Xout[i,5]/Xout[i,4])/10 * (1-Xout[i,11]), 
      sqrt(abs(Xout[i,7]*Xout[i,8]))/5,
      sqrt(abs(Xout[i,9]/Xout[i,6]))/3
    )
    samples = rWishart(1000, df = m + 1, Sigma = diag(D^2))
    samples_power = list()
    for (j in 1:(dim(samples)[3])) {
      sample_eigen = eigen(samples[,,j])
      P = sample_eigen$vectors
      Lambd_alpha = diag(pmax(0,sample_eigen$values)^0.5)
      samples_power[[j]] = P%*%Lambd_alpha%*%t(P)
    }
    
    LMeanOut_power[[i]] = Reduce("+", samples_power) / length(samples_power)
  }
  
  ################################### E2M ####################################
  layer = c(2,3,4,5,6)
  hidden = c(8,16,32,64,128)
  lamb = c(-1e-1,-1e-2,0,1e-2,1e-1)
  res_E2M = E2M(y = L, x = X, xout = Xout,
                optns = list(type = "covariance", metric = "power", alpha = 0.5,
                             layer = layer, hidden = hidden,
                             lamb = lamb, lr = 0.0005, dropout = 0.3,
                             axis = -2, num_epochs = 1000, seed = rrr))

  res_E2M_power = list()
  for (i in 1:nOut) {
    P = eigen(res_E2M$yPred[[i]])$vectors
    Lambd_alpha = diag(pmax(0,eigen(res_E2M$yPred[[i]])$values)^0.5)
    res_E2M_power[[i]] = P%*%Lambd_alpha%*%t(P)
  }
  err_E2M = sapply(1:length(LMeanOut_power), function(j) sum((res_E2M_power[[j]] - LMeanOut_power[[j]])^2))

  ##################################### GFR ####################################
  M = array(as.numeric(unlist(L)), dim=c(dim(L[[1]])[1],dim(L[[1]])[1],length(L)))
  gfr = frechet::GloCovReg(x = as.matrix(X), M = M, xout = as.matrix(Xout),
                           optns = list(metric = "power", alpha = 0.5))
  gfr_power = list()
  for (i in 1:nOut) {
    P = eigen(gfr$Mout[[i]])$vectors
    Lambd_alpha = diag(pmax(0,eigen(gfr$Mout[[i]])$values)**0.5)
    gfr_power[[i]] = P%*%Lambd_alpha%*%t(P)
  }
  err_gn = sapply(1:length(LMeanOut_power), function(j) sum((gfr_power[[j]] - LMeanOut_power[[j]])^2))

  ##################################### DFR ####################################
  res_dfr = DFR(y = L, x = X, xout = Xout,
                optns = list(type = "covariance", metric = "power", alpha = 0.5,
                             manifold = list(method = "isomap", k = n), r = 2,
                             layer = 4, hidden = 32, lr = 0.0005, dropout = 0.3,
                             num_epochs = 1000, seed = rrr))
  res_dfr_power = list()
  for (i in 1:nOut) {
    P = eigen(res_dfr$yPred[[i]])$vectors
    Lambd_alpha = diag(pmax(0,eigen(res_dfr$yPred[[i]])$values)^0.5)
    res_dfr_power[[i]] = P%*%Lambd_alpha%*%t(P)
  }
  err_dfr = sapply(1:length(LMeanOut_power), function(j) sum((res_dfr_power[[j]] - LMeanOut_power[[j]])^2))
  
  ################################# Random Forest ##############################
  X_forest = list()
  X_forest$type = "scalar"
  X_forest$id = 1:n
  X_forest$X = as.matrix(X)
  Y_forest = list()
  Y_forest$type = "covariance"
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
  X_new_forest$X = as.matrix(Xout)
  p = ncol(X)
  method = "Power"
  alpha = 1/2
  
  res_rfwlfr = rfwllfr(r = rrr, Scalar=X_forest, Y=Y_forest, Xout=X_new_forest, mtry=ceiling(4/5*p), deep=5, ntree=100, ncores=1)
  res_rfwlfr_power = list()
  for (i in 1:nOut) {
    P = eigen(res_rfwlfr$res[,,i])$vectors
    Lambd_alpha = diag(pmax(0,eigen(res_rfwlfr$res[,,i])$values)^0.5)
    res_rfwlfr_power[[i]] = P%*%Lambd_alpha%*%t(P)
  }
  err_RFWLFR = sapply(1:length(LMeanOut_power), 
                      function(j) sum((res_rfwlfr_power[[j]] - LMeanOut_power[[j]])^2))
  
  # Prediction errors
  err_q = data.frame(n = n, r = rrr,
                     E2M = mean(err_E2M),
                     DFR = mean(err_dfr),
                     GFR = mean(err_gn),
                     RFWLFR = mean(err_RFWLFR)
  )
  
  return(err_q)
}
# Stop the parallel cluster
stopCluster(cl)

# Table 1: SPD outputs with power metric
do.call(rbind, result)  %>%
  dplyr::select(-r) %>%
  gather(key = "method", value = Error, -c("n")) %>%
  group_by(n, method) %>%
  dplyr::summarise(MSPE = sprintf("%.4f", mean(Error)),
                   sd = sprintf("(%.4f)", sd(Error))) %>%
  gather(key = "measure", value = "value", MSPE:sd) %>%
  mutate(method = factor(method, levels = c("E2M", "DFR", "GFR", "RFWLFR"))) %>%
  dplyr::select(n, method, measure, value) %>%
  spread(key = method, value = value)
