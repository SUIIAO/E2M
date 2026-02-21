#' @title End-to-End Deep Learning for Predicting Metric Space-Valued Outputs (E2M)
#' @description Implements E2M for metric-space-valued responses with Euclidean predictors using various manifold learning techniques.
#' @param x An n by p matrix or data frame of predictors. Can also be a vector of length n if p = 1.
#' @param y A list of n observations, where each element represents the metric space-valued response.
#' @param xout Optional. An nOut by p matrix or data frame of output predictor levels for predictions. Default is \code{NULL}.
#' @param optns A list of options specified as \code{list(name = value)}. See `Details` for available control options.
#' @details The control options are:
#' \describe{
#'   \item{type}{The type of data ('measure' for probability measures, 'network' for network data, 'covariance' for covariance data, 'correlation' for correlation data, or 'SPD' for symmetric positive definite matrices). Required.}
#'   \item{layer}{Number of hidden layers for the neural network. Default is 4. Can be a list for CV.}
#'   \item{hidden}{Number of neurons per layer. Default is 64. Can be a list for CV.}
#'   \item{num_epochs}{Number of training epochs. Default is 2000.}
#'   \item{lr}{Learning rate for the neural network training. Default is 0.0005. Can be a list for CV.}
#'   \item{dropout}{Dropout rate for regularization. Default is 0.3. Can be a list for CV.}
#'   \item{lamb}{Lambda value for entropy penalty. Default is 0.0 (no penalty). Can be a list for CV.}
#'   \item{seed}{Random seed for reproducibility. If not specified, a random seed is generated.}
#'   \item{norm_axis}{Normalization axis for the neural network (-1 or -2). Default is -2.}
#'   \item{batch_size}{Batch size for training. Default is 32.}
#'   \item{n_anchor}{Number of anchor points to use for large datasets (n > 1000). Default is 1000.}
#'   \item{metric}{Metric for covariance/correlation/SPD data ('frobenius', 'power', or 'BW' for Bures-Wasserstein). Default 'frobenius'.}
#'   \item{alpha}{Power value if metric='power' for covariance/correlation data.}
#'   \item{digits}{Optional number of digits to round network projections.}
#'   \item{upper}{Optional upper bound for OSQP projection for 'measure' type.}
#'   \item{lower}{Optional lower bound for OSQP projection for 'measure' type.}
#' }
#' @return A list containing:
#' \item{yFit}{Fitted values for the training data (projected if applicable).}
#' \item{yPred}{Predicted values for the test data if \code{xout} is provided (projected if applicable).}
#' \item{best_params}{Best hyperparameters found via cross-validation (if performed).}
#' \item{final_err_train}{Training loss history for the final model.}
#' \item{final_err_valid}{Validation loss history for the final model.}
#' \item{min_final_valid_loss}{Minimum validation loss achieved by the final model.}
#' \item{nn_fit}{(Optional) A minimal Python NN bundle (state + settings) for post-hoc interpretability (e.g., permutation importance).}
#' \item{params_used}{A list of the actual parameters used for the final model run.}
#' @export
#' 
E2M = function(x, y, xout = NULL, optns = list()){
  
  # Check if inputs x and y are provided
  if (is.null(y) | is.null(x)) {
    stop("requires the input of both y and x")
  }
  
  # Check if the data type is specified
  if (is.null(optns$type)) {
    stop("requires the input of data type")
  }
  
  # Ensure x is a matrix
  if (!is.matrix(x)) {
    if (is.data.frame(x) | is.vector(x)) {
      x <- as.matrix(x)
    } else {
      stop("x must be a matrix or a data frame or a vector")
    }
  }
  
  n = nrow(x)  # Number of observations
  d = ncol(x)  # Number of predictor dimensions
  if(!is.null(xout)){
    nout = nrow(xout)
  }
  
  
  # Set random seed for reproducibility
  if (is.null(optns$seed)) {
    seed = rbinom(1, 1e3, 0.5)
    set.seed(seed)
  } else {
    seed = optns$seed
    set.seed(seed)
  }
  
  # Set neural network parameters with default values matching Python DWR_NN
  # Tunable parameters can now be lists/vectors
  layer      = if (!is.null(optns$layer)) optns$layer else 4
  hidden     = if (!is.null(optns$hidden)) optns$hidden else 64
  num_epochs = if (!is.null(optns$num_epochs)) optns$num_epochs else 2000
  lr         = if (!is.null(optns$lr)) optns$lr else 0.0005
  dropout    = if (!is.null(optns$dropout)) optns$dropout else 0.3
  lamb       = if (!is.null(optns$lamb)) optns$lamb else 0.0 # Default lamb to 0.0 (no penalty)
  batch_size = if (!is.null(optns$batch_size)) optns$batch_size else 32
  norm_axis  = if (!is.null(optns$norm_axis)) optns$norm_axis else -2L # Default to -2 (integer)
  if(!is.null(optns$n_anchor)){
    n_anchor = optns$n_anchor
  }else if(n>2000){
    warning("Large n detected (n > 2000). Consider setting 'n_anchor' in optns for efficiency.")
    n_anchor = n
  }else{
    n_anchor = n
  }
  
  if (!is.null(optns$metric)){
    metric = optns$metric
  } else {
    stop("requires the input of metric for the metric space.")
  }

  # Optional: Add warnings if default values are used for tunable params (optional)
  # if (is.null(optns$layer)) warning("Hidden layer not specified, using default: 4")
  # ... add similar warnings if desired ...

  if(optns$type == "network"){
    # Prepare y_ based on metric, similar to covariance/correlation
    if(!is.null(optns$metric) && optns$metric == "power"){
      # Power metric transformation for network data
      alpha = optns$alpha
      # Need to determine dimensions correctly for network (assuming square matrices)
      m_dim = dim(y[[1]])[1] 
      M = array(as.numeric(unlist(y)), dim=c(m_dim, m_dim, length(y)))
      M_power = array(0, dim=c(m_dim, m_dim, length(y)))
      if(alpha > 0){
        for(i in 1:n){ # Use n (number of observations) defined earlier
          # Use tryCatch for robustness against non-positive definite matrices if necessary
          eigen_decomp <- eigen(M[,,i])
          P <- eigen_decomp$vectors
          Lambd_alpha <- diag(pmax(0, eigen_decomp$values)**alpha)
          M_power[,,i] <- P %*% Lambd_alpha %*% t(P)
        }
      }
      # Flatten the transformed matrices
      y_ = matrix(M_power, nrow = n, ncol = m_dim*m_dim, byrow = TRUE)
      
    } else {
      # Default: Flatten network matrices directly (Frobenius norm equivalent)
      y_ = do.call(rbind, lapply(y, function(i) as.numeric(i)))
    }
  }else if(optns$type == "measure"){
    N = sapply(y, length)
    y = lapply(1:n, function(i) {
      sort(y[[i]])
    })
    M = plcm(N) # least common multiple of N_i
    yM = t(sapply(1:n, function(i) {
      rep(y[[i]], each = M / N[i])
    })) # n by M
    
    M = N[1] # least common multiple of N_i
    y_ = matrix(unlist(y),nrow = length(y), byrow = T)
    
  }else if(optns$type %in% c("covariance", "correlation", "SPD")){
    # Use short-circuiting OR (||) to avoid logical(0) error when optns$metric is NULL
    if(is.null(optns$metric) || optns$metric == "frobenius" || optns$metric == "BW"){
      # by default, using Frobenius norm
      y_ = do.call(rbind, lapply(y, function(i) as.numeric(i)))
      # Use short-circuiting OR (||) here too, although the first condition likely covers it
    }else if(!is.null(optns$metric) && optns$metric == "power"){ # Check for non-NULL *and* equality
      # Power metric
      alpha = optns$alpha
      M = array(as.numeric(unlist(y)), dim=c(dim(y[[1]])[1],dim(y[[1]])[1],length(y)))
      M_power = array(0, dim=c(dim(y[[1]])[1],dim(y[[1]])[1],length(y)))
      if(alpha>0){
        for(i in 1:n){
          P = eigen(M[,,i])$vectors
          Lambd_alpha = diag(pmax(0,eigen(M[,,i])$values)**alpha)
          M_power[,,i] = P%*%Lambd_alpha%*%t(P)
        }
      }
      y_ = matrix(M_power, nrow = dim(M_power)[3], 
                  ncol = dim(M_power)[1]*dim(M_power)[2],
                  byrow = TRUE)
    }
  }else if(optns$type %in% c("scalar")){
    y_ = matrix(unlist(y), n, 1)
  }

  res_E2M = E2M_NN(x = x, 
                       y = y_, 
                       xout = xout, 
                       data_type = optns$type,
                       lamb = lamb,           # Pass potentially list variable
                       layer = layer,         # Pass potentially list variable
                       hidden = hidden,       # Pass potentially list variable
                       dropout = dropout,     # Pass potentially list variable
                       lr = lr,             # Pass potentially list variable
                       num_epochs = num_epochs, # Pass variable
                       seed = seed,           # Pass variable
                       batch_size = batch_size, # Pass variable
                       norm_axis = norm_axis,    # Pass norm_axis variable
                       n_anchor = n_anchor, # Pass n_anchor variable
                       metric = metric
  )
  
  # Projection
  if(optns$type == "measure"){
    # initialization of OSQP solver
    A <- cbind(diag(M), rep(0, M)) + cbind(rep(0, M), -diag(M))
    if (!is.null(optns$upper) &
        !is.null(optns$lower)) {
      # if lower & upper are neither NULL
      l <- c(optns$lower, rep(0, M - 1), -optns$upper)
    } else if (!is.null(optns$upper)) {
      # if lower is NULL
      A <- A[, -1]
      l <- c(rep(0, M - 1), -optns$upper)
    } else if (!is.null(optns$lower)) {
      # if upper is NULL
      A <- A[, -ncol(A)]
      l <- c(optns$lower, rep(0, M - 1))
    } else {
      # if both lower and upper are NULL
      A <- A[, -c(1, ncol(A))]
      l <- rep(0, M - 1)
    }
    # P <- as(diag(M), "sparseMatrix")
    # A <- as(t(A), "sparseMatrix")
    P <- diag(M)
    A <- t(A)
    q <- rep(0, M)
    u <- rep(Inf, length(l))
    model <-
      osqp::osqp(
        P = P,
        q = q,
        A = A,
        l = l,
        u = u,
        osqp::osqpSettings(verbose = FALSE)
      )
    yFit = matrix(0, n, M)
    yFitSup = 1:M / M 
    
    for (i in 1:n) {
      qNew <- res_E2M$y_fit[i,]
      model$Update(q = -qNew)
      yFit[i, ] <- sort(model$Solve()$x)
    }
    if(!is.null(xout)){
      yPred = matrix(0, nout, M)
      yPredSup = 1:M/M 
      for (i in 1:nout) {
        yPred[i,] <- res_E2M$y_pred[i,]
      }
    }
    
  }else if(optns$type == "network"){
    m = dim(y[[1]])[1]
    
    # Check if power metric is used, requiring OSQP projection
    if (!is.null(optns$metric) && optns$metric == "power") {
      # --- OSQP Setup for Network Power Metric --- 
      W = 2^32 # bound on the weights of edges in the graph
      nConsts = m^2 # number of constraints
      l = c(rep.int(0, m * (m + 1) / 2), rep.int(-W, m * (m - 1) / 2))
      u = rep.int(0, nConsts)
      q = rep.int(0, m^2)
      P = diag(m^2)
      consts = matrix(0, nrow = nConsts, ncol = m^2)
      k = 0
      # symmetric constraints
      for (i in 1:(m - 1)) {
        for (j in (i + 1):m) {
          k = k + 1
          consts[k, (j - 1) * m + i] = 1
          consts[k, (i - 1) * m + j] = -1
        }
      }
      for (i in 1:m) {
        consts[k + i, ((i - 1) * m + 1):(i * m)] = rep(1, m)
      }
      k = k + m
      for (i in 1:(m - 1)) {
        for (j in (i + 1):m) {
          k = k + 1
          consts[k, (j - 1) * m + i] = 1
        }
      }
      model <- osqp::osqp(P, q, consts, l, u, osqp::osqpSettings(verbose = FALSE))
      # --- End OSQP Setup ---
      
      yFit = list()
      nodes <- colnames(y[[1]]) # Get node names once
      if (is.null(nodes)) nodes <- 1:m
      
      for (i in 1:n) {
        # --- OSQP Projection for yFit (Power Metric) ---
        qNew = res_E2M$y_fit[i,]
        model$Update(q = -qNew)
        temp <- matrix(model$Solve()$x, ncol = m, dimnames = list(nodes, nodes))
        temp <- (temp + t(temp)) / 2 # symmetrize
        if (!is.null(optns$digits)) temp <- round(temp, digits = optns$digits) # round
        temp[temp > 0] <- 0 # off diagonal should be negative
        diag(temp) <- 0
        diag(temp) <- -colSums(temp)
        yFit[[i]] <- temp
        # --- End OSQP Projection ---
      }
      
      if(!is.null(xout)){
        yPred = list()
        for (i in 1:nout) {
          # --- OSQP Projection for yPred (Power Metric) ---
          qNew = res_E2M$y_pred[i,]
          model$Update(q = -qNew)
          temp <- matrix(model$Solve()$x, ncol = m, dimnames = list(nodes, nodes))
          temp <- (temp + t(temp)) / 2 # symmetrize
          if (!is.null(optns$digits)) temp <- round(temp, digits = optns$digits) # round
          temp[temp > 0] <- 0 # off diagonal should be negative
          diag(temp) <- 0
          diag(temp) <- -colSums(temp)
          yPred[[i]] <- temp
          # --- End OSQP Projection ---
        }
      } else {
        yPred = NULL # Ensure yPred is defined if xout is NULL
      }
      
    } else { # Default case: No OSQP projection for network (metric != 'power')
      yFit = list()
      for (i in 1:n) {
        yFit[[i]] <- matrix(res_E2M$y_fit[i,], m, m)
      }
      
      if(!is.null(xout)){
        yPred = list()
        for (i in 1:nout) {
          yPred[[i]] <- matrix(res_E2M$y_pred[i,], m, m)
        }
      } else {
        yPred = NULL # Ensure yPred is defined if xout is NULL
      }
    } # End if/else for optns$metric == "power"
    
  } else if(optns$type == "covariance"){
    m = dim(y[[1]])[1]
    yFit = list()
    for (i in 1:n) {
      M = matrix(res_E2M$y_fit[i,],m,m)
      # Check metric only if it's not NULL
      if(!is.null(optns$metric) && optns$metric == "power"){
        P = eigen(M)$vectors
        Lambd_alpha = diag(pmax(0,eigen(M)$values)**(1/alpha))
        M = P%*%Lambd_alpha%*%t(P)
        M = as.matrix(Matrix::forceSymmetric(M))
      }
      M = as.matrix(Matrix::nearPD(M,corr = FALSE)$mat)
      yFit[[i]] = M
    }
    if(!is.null(xout)){
      yPred = list()
      for (i in 1:nout) {
        M = matrix(as.numeric(res_E2M$y_pred[i,]),m,m)
        # Check metric only if it's not NULL
        if(!is.null(optns$metric) && optns$metric == "power"){
          P = eigen(M)$vectors
          Lambd_alpha = diag(pmax(0,eigen(M)$values)**(1/alpha))
          M = P%*%Lambd_alpha%*%t(P)
          M = as.matrix(Matrix::forceSymmetric(M))
        }
        M = as.matrix(Matrix::nearPD(M,corr = FALSE)$mat)
        yPred[[i]] = M
      }
    }
  } else if(optns$type == "correlation"){
    m = dim(y[[1]])[1]
    yFit = list()
    for (i in 1:n) {
      M = matrix(res_E2M$y_fit[i,],m,m)
      # Check metric only if it's not NULL
      if(!is.null(optns$metric) && optns$metric == "power"){
        P = eigen(M)$vectors
        Lambd_alpha = diag(pmax(0,eigen(M)$values)**(1/alpha))
        M = P%*%Lambd_alpha%*%t(P)
        M = as.matrix(Matrix::forceSymmetric(M))
      }
      M = as.matrix(Matrix::nearPD(M,corr = TRUE)$mat)
      yFit[[i]] = M
    }
    if(!is.null(xout)){
      yPred = list()
      for (i in 1:nout) {
        M = matrix(as.numeric(res_E2M$y_pred[i,]),m,m)
        # Check metric only if it's not NULL
        if(!is.null(optns$metric) && optns$metric == "power"){
          P = eigen(M)$vectors
          Lambd_alpha = diag(pmax(0,eigen(M)$values)**(1/alpha))
          M = P%*%Lambd_alpha%*%t(P)
          M = as.matrix(Matrix::forceSymmetric(M))
        }
        M = as.matrix(Matrix::nearPD(M,corr = FALSE)$mat)
        yPred[[i]] = M
      }
    }
  } else if(optns$type == "scalar"){
    yFit = matrix(res_E2M$y_fit, n, 1)

    if(!is.null(xout)){
      yPred = matrix(res_E2M$y_pred, nout, 1)
    }
  } else if(optns$type == "SPD"){
    # For SPD matrices, reshape the flattened results back to matrix form
    m = dim(y[[1]])[1]
    yFit = list()
    for (i in 1:n) {
      yFit[[i]] <- matrix(res_E2M$y_fit[i,], m, m)
    }
    
    if(!is.null(xout)){
      yPred = list()
      for (i in 1:nout) {
        yPred[[i]] <- matrix(res_E2M$y_pred[i,], m, m)
      }
    }
  }
  
  # --- Prepare Results --- 
  results <- list(
    yFit = yFit,
    best_params = res_E2M$best_params,          # From Python result
    final_err_train = res_E2M$final_err_train,  # From Python result
    final_err_valid = res_E2M$final_err_valid,  # From Python result
    min_final_valid_loss = res_E2M$min_final_valid_loss, # From Python result
    use_anchor_sampling = res_E2M$use_anchor_sampling,   # From Python result
    n_anchor_used = res_E2M$n_anchor_used,               # From Python result
    anchor_indices = res_E2M$anchor_indices,             # From Python result
    # Minimal NN bundle for post-hoc interpretability from R:
    # This is intentionally *not* the whole fitted python object, just what permutation importance needs.
    nn_fit = list(
      nn_state_dict_cpu = res_E2M$nn_state_dict_cpu,
      nn_model_settings_cpu = res_E2M$nn_model_settings_cpu
    ),
    params_used = list(                        # R parameters passed to Python
      lamb = lamb, layer = layer, hidden = hidden,
      dropout = dropout, lr = lr, num_epochs = num_epochs, seed = seed,
      batch_size = batch_size, norm_axis = norm_axis, # Added norm_axis for completeness
      n_anchor = n_anchor # Added n_anchor for completeness
    )
  )
  
  # Add type-specific or conditional outputs
  if (optns$type == "measure") {
    results$yFitSup <- yFitSup # Add support points for measure type
  }
  
  if (!is.null(xout)) { # If prediction points were provided
    results$yPred <- yPred
    if (optns$type == "measure") {
      results$yPredSup <- yPredSup # Add prediction support points for measure type
    }
  }
  
  # Return the consolidated list
  return(results)
}

#' @title Permutation Importance for E2M Neural Network
#' @description Compute permutation feature importance for a fitted E2M neural network by measuring the increase in prediction loss after permuting each predictor.
#' @param fit An object returned by \code{E2M()} (must contain \code{fit$nn_fit}).
#' @param x An n by p matrix/data.frame of predictors used for evaluation (typically a test set).
#' @param y Response observations for evaluation. Can be either:
#'   (i) a list of n objects (same format as \code{E2M()} input), or
#'   (ii) an n by q numeric matrix of flattened responses (already preprocessed).
#' @param optns A list of options specified as \code{list(name = value)}. Required fields:
#'   \describe{
#'     \item{type}{Data type (same as \code{E2M()}), used only when \code{y} is a list.}
#'     \item{metric}{Metric for covariance/correlation/SPD (used only when \code{y} is a list).}
#'     \item{alpha}{Power value if \code{metric='power'} (used only when \code{y} is a list).}
#'     \item{n_repeats}{Number of permutation repeats per feature. Default 20.}
#'     \item{seed}{Seed for permutation RNG. Default 123.}
#'     \item{batch_size}{Batch size for evaluation. Default 512.}
#'     \item{feature_names}{Optional character vector of length p.}
#'   }
#' @return A list with baseline loss and importance results (mean/std and raw repeats), sorted by decreasing importance.
#' @export
E2M_permutation_importance = function(fit, x, y, optns = list()){
  if (is.null(fit) || is.null(fit$nn_fit)) {
    stop("`fit` must be the output of E2M() and contain `fit$nn_fit`.")
  }
  if (is.null(x)) stop("`x` is required.")

  # Ensure x is a matrix
  if (!is.matrix(x)) {
    if (is.data.frame(x) | is.vector(x)) {
      x <- as.matrix(x)
    } else {
      stop("x must be a matrix or a data frame or a vector")
    }
  }
  n = nrow(x)

  # If y is already numeric matrix, use it directly; otherwise preprocess like E2M()
  if (is.list(y)) {
    if (is.null(optns$type)) stop("When y is a list, optns$type is required for preprocessing.")
    if (is.null(optns$metric)) stop("When y is a list, optns$metric is required for preprocessing.")

    if(optns$type == "network"){
      if(!is.null(optns$metric) && optns$metric == "power"){
        alpha = optns$alpha
        m_dim = dim(y[[1]])[1]
        M = array(as.numeric(unlist(y)), dim=c(m_dim, m_dim, length(y)))
        M_power = array(0, dim=c(m_dim, m_dim, length(y)))
        if(alpha > 0){
          for(i in 1:n){
            eigen_decomp <- eigen(M[,,i])
            P <- eigen_decomp$vectors
            Lambd_alpha <- diag(pmax(0, eigen_decomp$values)**alpha)
            M_power[,,i] <- P %*% Lambd_alpha %*% t(P)
          }
        }
        y_ = matrix(M_power, nrow = n, ncol = m_dim*m_dim, byrow = TRUE)
      } else {
        y_ = do.call(rbind, lapply(y, function(i) as.numeric(i)))
      }
    } else if(optns$type == "measure"){
      y_ = matrix(unlist(y), nrow = length(y), byrow = TRUE)
    } else if(optns$type %in% c("covariance", "correlation", "SPD")){
      if(is.null(optns$metric) || optns$metric == "frobenius" || optns$metric == "BW"){
        y_ = do.call(rbind, lapply(y, function(i) as.numeric(i)))
      } else if(!is.null(optns$metric) && optns$metric == "power"){
        alpha = optns$alpha
        M = array(as.numeric(unlist(y)), dim=c(dim(y[[1]])[1],dim(y[[1]])[1],length(y)))
        M_power = array(0, dim=c(dim(y[[1]])[1],dim(y[[1]])[1],length(y)))
        if(alpha>0){
          for(i in 1:n){
            P = eigen(M[,,i])$vectors
            Lambd_alpha = diag(pmax(0,eigen(M[,,i])$values)**alpha)
            M_power[,,i] = P%*%Lambd_alpha%*%t(P)
          }
        }
        y_ = matrix(M_power, nrow = dim(M_power)[3],
                    ncol = dim(M_power)[1]*dim(M_power)[2],
                    byrow = TRUE)
      }
    } else if(optns$type %in% c("scalar")){
      y_ = matrix(unlist(y), n, 1)
    } else {
      stop(paste0("Unsupported type for preprocessing y list: ", optns$type))
    }
  } else {
    # Assume y is already flattened numeric matrix/vector
    y_ = y
    if (is.vector(y_)) y_ = matrix(y_, ncol = 1)
  }

  n_repeats = if(!is.null(optns$n_repeats)) as.integer(optns$n_repeats) else 100L
  # Reticulate may pass R numeric as python float (e.g., 1.0); force integer seed for NumPy.
  seed = if(!is.null(optns$seed)) as.integer(optns$seed) else 123L
  batch_size = if(!is.null(optns$batch_size)) optns$batch_size else 512L
  feature_names = if(!is.null(optns$feature_names)) optns$feature_names else NULL

  # Call Python permutation importance (assumes E2M.py has been sourced via reticulate)
  res = permutation_importance_E2M(
    fit = fit$nn_fit,
    x = x,
    y = y_,
    n_repeats = n_repeats,
    seed = seed,
    batch_size = batch_size,
    feature_names = feature_names
  )

  return(res)
}

#' @title Bures-Wasserstein Barycenter for SPD Matrices
#' @description Computes the Bures-Wasserstein barycenter of a collection of SPD matrices using the fixed-point iteration algorithm.
#' @param spd_matrices A list of SPD matrices (each matrix should be m x m)
#' @param weights A vector of positive weights corresponding to each matrix (should sum to 1)
#' @param max_iter Maximum number of iterations for the fixed-point algorithm. Default is 15.
#' @param eps Small value for numerical stability. Default is 1e-6.
#' @return The barycenter matrix (m x m SPD matrix)
#' @details This function implements the fixed-point iteration algorithm from the paper:
#' S^(k+1) = S^(k)^(-1/2) * (sum_i w_i * (S^(k)^(1/2) * A_i * S^(k)^(1/2))^(1/2))^2 * S^(k)^(-1/2)
#' 
#' The algorithm starts with the identity matrix as the initial guess and iterates until convergence
#' or the maximum number of iterations is reached.
#' @export
BW_barycenter <- function(spd_matrices, weights, max_iter = 10, eps = 1e-6) {
  
  # Input validation
  if (length(spd_matrices) != length(weights)) {
    stop("Number of matrices must equal number of weights")
  }
  
  if (abs(sum(weights) - 1.0) > 1e-10) {
    warning("Weights do not sum to 1, normalizing...")
    weights <- weights / sum(weights)
  }
  
  if (any(weights < 0)) {
    stop("All weights must be non-negative")
  }
  
  # Get matrix dimension
  m <- nrow(spd_matrices[[1]])
  n <- length(spd_matrices)
  
  # Check that all matrices are square and same size
  for (i in 1:n) {
    if (nrow(spd_matrices[[i]]) != m || ncol(spd_matrices[[i]]) != m) {
      stop(paste("Matrix", i, "is not", m, "x", m))
    }
  }
  
  # Initialize barycenter S^(0) as identity matrix (as suggested in the paper)
  S_current <- diag(m)
  
  # Fixed-point iteration
  for (iteration in 1:max_iter) {
    
    # Compute S^(k)^(1/2) and S^(k)^(-1/2) for current iterate
    eig_S <- eigen(S_current, symmetric = TRUE)
    eigenvals_S <- pmax(eig_S$values, eps)  # Clamp to ensure positivity
    eigenvecs_S <- eig_S$vectors
    
    sqrt_eigenvals_S <- sqrt(eigenvals_S)
    inv_sqrt_eigenvals_S <- 1.0 / sqrt_eigenvals_S
    
    sqrt_S <- eigenvecs_S %*% diag(sqrt_eigenvals_S) %*% t(eigenvecs_S)
    inv_sqrt_S <- eigenvecs_S %*% diag(inv_sqrt_eigenvals_S) %*% t(eigenvecs_S)
    
    # Compute weighted sum: sum_i w_i * (S^(k)^(1/2) * A_i * S^(k)^(1/2))^(1/2)
    weighted_sum <- matrix(0, nrow = m, ncol = m)
    
    for (i in 1:n) {
      # Get weight for matrix A_i
      w_i <- weights[i]
      A_i <- spd_matrices[[i]]
      
      # Compute S^(k)^(1/2) * A_i * S^(k)^(1/2)
      sandwich <- sqrt_S %*% A_i %*% sqrt_S
      
      # Compute (S^(k)^(1/2) * A_i * S^(k)^(1/2))^(1/2)
      eig_sandwich <- eigen(sandwich, symmetric = TRUE)
      eigenvals_sandwich <- pmax(eig_sandwich$values, eps)  # Clamp to ensure positivity
      eigenvecs_sandwich <- eig_sandwich$vectors
      
      sqrt_eigenvals_sandwich <- sqrt(eigenvals_sandwich)
      sqrt_sandwich <- eigenvecs_sandwich %*% diag(sqrt_eigenvals_sandwich) %*% t(eigenvecs_sandwich)
      
      # Add weighted contribution
      weighted_sum <- weighted_sum + w_i * sqrt_sandwich
    }
    
    # Square the weighted sum: (sum_i w_i * (S^(k)^(1/2) * A_i * S^(k)^(1/2))^(1/2))^2
    squared_sum <- weighted_sum %*% weighted_sum
    
    # Apply the transformation: S^(k+1) = S^(k)^(-1/2) * squared_sum * S^(k)^(-1/2)
    S_next <- inv_sqrt_S %*% squared_sum %*% inv_sqrt_S
    
    # Check for convergence
    if (iteration > 1) {
      diff <- norm(S_next - S_current, type = "F")
      if (diff < eps) {
        break
      }
    }
    
    S_current <- S_next
  }
  
  # Ensure the result is symmetric (numerical precision)
  S_current <- (S_current + t(S_current)) / 2
  
  return(S_current)
}
