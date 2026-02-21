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

# Function to compute Bures-Wasserstein distance between two SPD matrices
bures_wasserstein_distance <- function(A, B) {
  # Compute tr(A) + tr(B)
  tr_A <- sum(diag(A))
  tr_B <- sum(diag(B))
  
  # Compute A^(1/2)
  eig_A <- eigen(A)
  sqrt_A <- eig_A$vectors %*% diag(sqrt(pmax(eig_A$values, 1e-10))) %*% t(eig_A$vectors)
  
  # Compute A^(1/2) * B * A^(1/2)
  sqrt_A_B_sqrt_A <- sqrt_A %*% B %*% sqrt_A
  
  # Compute (A^(1/2) * B * A^(1/2))^(1/2)
  eig_AB <- eigen(sqrt_A_B_sqrt_A)
  sqrt_eig_AB <- sqrt(pmax(eig_AB$values, 1e-10))
  tr_sqrt_AB <- sum(sqrt_eig_AB)
  
  # Bures-Wasserstein distance: sqrt(tr(A) + tr(B) - 2*tr((A^(1/2)*B*A^(1/2))^(1/2)))
  bw_dist_squared <- tr_A + tr_B - 2 * tr_sqrt_AB
  bw_dist <- sqrt(pmax(bw_dist_squared, 1e-10))
  
  return(bw_dist)
}

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
  reticulate::source_python('../code/E2M/E2M.py')
  
  nOut = 200 # Number of test samples
  m = 2
  
  set.seed(rrr) # Set seed for reproducibility
  
  # Generate predictor data for training and test sets
  X = data.frame(
    X1 = runif(n,0,1), X2 = runif(n,-1/2,1/2), X3 = runif(n,1,2),
    X4 = sample(c(0,1),n,TRUE,c(0.4,0.6)),
    X5 = sample(c(0,1),n,TRUE,c(0.5,0.5))
  )
  Xout = data.frame(
    X1 = runif(nOut,0,1), X2 = runif(nOut,-1/2,1/2), X3 = runif(nOut,1,2),
    X4 = sample(c(0,1),nOut,TRUE,c(0.4,0.6)),
    X5 = sample(c(0,1),nOut,TRUE,c(0.5,0.5))
  )
  
  # Generate response data
  L = list()
  for(i in 1:n){
    D = c(
      sin(X[i,1]*pi)*X[i,4] + cos(X[i,2]*pi)*(1-X[i,4]), 
      sin(X[i,2]*pi)*cos(X[i,3]*pi)
    )
    
    L[[i]] =  rWishart(1, df = m + 1, Sigma = diag(D^2))[,,1]
  }
  
  # Generate true quantile values for test data
  LMeanOut = list()
  for(i in 1:nOut){
    D = c(
      sin(Xout[i,1]*pi)*Xout[i,4] + cos(Xout[i,2]*pi)*(1-Xout[i,4]), 
      sin(Xout[i,2]*pi)*cos(Xout[i,3]*pi)
    )
    
    sample_i = rWishart(500, df = m + 1, Sigma = diag(D^2))
    sample_i = lapply(1:(dim(sample_i)[3]), function(j) sample_i[,,j])
    LMeanOut[[i]] = BW_barycenter(spd_matrices = sample_i, weights = rep(1/length(sample_i), length(sample_i)))
  }
  
  ################################### E2M ####################################
  layer = c(2,3,4,5,6)
  hidden = c(8,16,32,64,128)
  lamb = c(-1e-1,-1e-2,0,1e-2,1e-1)
  res_E2M = E2M(y = L, x = X, xout = Xout,
                optns = list(type = "SPD", metric = "BW",
                             layer = layer, hidden = hidden, 
                             lamb = lamb, lr = 0.005, dropout = 0.3, 
                             batch_size = n, axis = -2, num_epochs = 500, seed = rrr))
  
  # Calculate Bures-Wasserstein distance between predicted and true SPD matrices
  err_E2M = sapply(1:length(LMeanOut), function(j) {
    bures_wasserstein_distance(res_E2M$yPred[[j]], LMeanOut[[j]])
  })
  
  # Prediction errors
  err_q = data.frame(n = n, r = rrr,
                     E2M = mean(err_E2M)
  )
  
  return(err_q)
}
# Stop the parallel cluster
stopCluster(cl)



# Table 1: SPD outputs with Bures-Wasserstein metric
do.call(rbind, result)  %>%
  dplyr::select(-r) %>%
  gather(key = "method", value = Error, -c("n")) %>%
  group_by(n, method) %>%
  dplyr::summarise(MSPE = sprintf("%.4f", mean(Error)),
                   sd = sprintf("(%.4f)", sd(Error))) %>%
  gather(key = "measure", value = "value", MSPE:sd) %>%
  mutate(method = factor(method, levels = c("E2M"))) %>%
  dplyr::select(n, method, measure, value) %>%
  spread(key = method, value = value)
