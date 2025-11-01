# PCR Functions - Parallel to LASSO structure
# Method 2: PCA and regression performed together in each window

# Required packages
# library(glmnet)  # For LASSO (already in your code)
# No additional packages needed for basic PCR - using base R prcomp()

#' Function 1: Train PCR on first n-1 rows, predict last row with specified number of components
#'
#' @param Y Data matrix with all variables
#' @param indice Index of the response variable column
#' @param ncomp Number of principal components to use
#' @return List containing model info, prediction, loadings, and PC scores
runpcr <- function(Y, indice, ncomp) {
  
  # Validate inputs
  if (nrow(Y) < 2) {
    stop("Y must have at least 2 rows")
  }
  if (indice < 1 || indice > ncol(Y)) {
    stop("indice must be a valid column index")
  }
  if (ncomp < 1) {
    stop("ncomp must be at least 1")
  }
  
  # Extract training data (all rows except the last)
  y_train <- Y[1:(nrow(Y)-1), indice]
  X_train <- Y[1:(nrow(Y)-1), -indice, drop = FALSE]
  
  # Extract test data (only the last row)
  X_test <- Y[nrow(Y), -indice, drop = FALSE]
  
  # Check that ncomp doesn't exceed available dimensions
  max_comp <- min(nrow(X_train) - 1, ncol(X_train))
  if (ncomp > max_comp) {
    warning(paste("ncomp reduced from", ncomp, "to", max_comp, 
                  "(maximum available components)"))
    ncomp <- max_comp
  }
  
  # Perform PCA on training data (scale and centre)
  pca_model <- prcomp(X_train, scale. = TRUE, center = TRUE)
  
  # Extract first ncomp principal components (scores)
  pc_train <- pca_model$x[, 1:ncomp, drop = FALSE]
  
  # Regress y on the principal components
  # Using lm for simplicity and to get standard regression output
  pc_df <- data.frame(y = y_train, pc_train)
  lm_model <- lm(y ~ ., data = pc_df)
  
  # Transform test data using training PCA
  # Centre and scale using training means and SDs
  X_test_scaled <- scale(X_test, 
                         center = pca_model$center, 
                         scale = pca_model$scale)
  
  # Project onto principal components
  pc_test <- X_test_scaled %*% pca_model$rotation[, 1:ncomp, drop = FALSE]
  
  # Predict using the linear model
  pc_test_df <- data.frame(pc_test)
  colnames(pc_test_df) <- colnames(pc_train)
  pred <- predict(lm_model, newdata = pc_test_df)
  
  # Return results
  return(list(
    "lm_model" = lm_model,           # Linear model fitted on PCs
    "pca_model" = pca_model,         # PCA model (contains loadings)
    "pred" = as.numeric(pred),       # Prediction
    "loadings" = pca_model$rotation[, 1:ncomp, drop = FALSE],  # PC loadings
    "ncomp_used" = ncomp             # Number of components actually used
  ))
}


#' Function 2: Rolling window forecasting with specified number of components (with date support)
#'
#' @param Y Data matrix with all variables
#' @param nprev Number of observations to test for
#' @param indice Index of the response variable column
#' @param ncomp Number of principal components to use
#' @param date_col Optional date column for plotting
#' @return List containing predictions, loadings over time, and errors
pcr.rolling.window <- function(Y, nprev, indice = 1, ncomp, date_col = NULL) {
  
  # Initialize storage
  n_predictors <- ncol(Y) - 1
  save.loadings <- array(NA, dim = c(nprev, n_predictors, ncomp))
  save.pred <- matrix(NA, nprev, 1)
  save.ncomp.used <- numeric(nprev)
  
  # Rolling window loop
  for(i in nprev:1) {
    Y.window <- Y[(1 + nprev - i):(1 + nrow(Y) - i), ]
    pcr.model <- runpcr(Y.window, indice, ncomp = ncomp)
    
    # Save results
    save.pred[(1 + nprev - i), ] <- pcr.model$pred
    save.loadings[(1 + nprev - i), , ] <- pcr.model$loadings
    save.ncomp.used[(1 + nprev - i)] <- pcr.model$ncomp_used
  }
  
  # Get actual values
  real <- Y[, indice]
  real_test <- tail(real, nprev)
  
  # Handle dates if provided
  if (!is.null(date_col)) {
    dates <- tail(date_col, nprev)
    
    # Convert YYYYMM to Date format for better plotting
    if (is.numeric(dates)) {
      # Assume format is YYYYMM (e.g., 202301)
      year <- floor(dates / 100)
      month <- dates %% 100
      dates <- as.Date(paste(year, month, "01", sep = "-"))
    } else if (!inherits(dates, "Date")) {
      # Try to convert to Date if not already
      dates <- as.Date(as.character(dates), format = "%Y%m")
    }
    
    # Plot with dates on x-axis
    plot(dates, real_test, type = "l", main = paste("PCR: ncomp =", ncomp),
         ylab = "Value", xlab = "Date", xaxt = "n")
    lines(dates, save.pred, col = "blue")
    
    # Show dates at regular intervals
    n_labels <- 10  # Adjust this number
    label_indices <- seq(1, length(dates), length.out = n_labels)
    axis.Date(1, at = dates[label_indices], format = "%Y-%m")
    
  } else {
    # Plot without dates (original behaviour)
    plot(real_test, type = "l", main = paste("PCR: ncomp =", ncomp),
         ylab = "Value", xlab = "Time Period")
    lines(save.pred, col = "blue")
  }
  
  legend("topright", legend = c("Actual", "Predicted"), 
         col = c("black", "blue"), lty = 1)
  
  # Compute errors
  rmse <- sqrt(mean((real_test - save.pred)^2))
  mae <- mean(abs(real_test - save.pred))
  errors <- c("rmse" = rmse, "mae" = mae)
  
  return(list(
    "pred" = save.pred, 
    "loadings" = save.loadings,  # 3D array: [time, predictors, components]
    "ncomp_used" = save.ncomp.used,
    "errors" = errors
  ))
}


#' Function 3: Cross-validation to find optimal number of components
#'
#' @param Y Data matrix with all variables
#' @param nprev Number of observations to test for
#' @param indice Index of the response variable column
#' @param ncomp_grid Vector of component numbers to test (default: 1 to min(4, sqrt(p)))
#' @param date_col Optional date column for plotting
#' @return List containing best ncomp, best model, and CV results

pcr.cv.ncomp <- function(Y, nprev, indice = 1, 
                         ncomp_grid = NULL,
                         date_col = NULL) {
  
  # Determine default ncomp_grid if not provided
  if (is.null(ncomp_grid)) {
    n_predictors <- ncol(Y) - 1
    max_ncomp <- min(4, floor(sqrt(n_predictors)))
    ncomp_grid <- 1:max_ncomp
    cat("Using default ncomp_grid: 1 to", max_ncomp, "\n")
  }
  
  # Validate ncomp_grid
  n_predictors <- ncol(Y) - 1
  max_possible <- min(nrow(Y) - nprev - 2, n_predictors)
  if (max(ncomp_grid) > max_possible) {
    warning(paste("Maximum ncomp in grid reduced from", max(ncomp_grid), 
                  "to", max_possible))
    ncomp_grid <- ncomp_grid[ncomp_grid <= max_possible]
  }
  
  # Store results for each ncomp
  cv_results <- data.frame(
    ncomp = ncomp_grid,
    rmse = NA,
    mae = NA
  )
  
  cat("Testing", length(ncomp_grid), "component values...\n")
  
  # Test each ncomp value
  for(i in seq_along(ncomp_grid)) {
    ncomp_val <- ncomp_grid[i]
    
    # Run rolling window with this number of components
    result <- pcr.rolling.window(Y, nprev, indice, ncomp = ncomp_val, 
                                 date_col = date_col)
    
    # Store errors
    cv_results$rmse[i] <- result$errors["rmse"]
    cv_results$mae[i] <- result$errors["mae"]
    
    cat("Completed ncomp =", ncomp_val, "- RMSE:", 
        round(result$errors["rmse"], 4), "\n")
  }
  
  # Find best ncomp (minimum RMSE)
  best_idx <- which.min(cv_results$rmse)
  best_ncomp <- cv_results$ncomp[best_idx]
  
  cat("\nBest ncomp:", best_ncomp, "with RMSE:", 
      round(cv_results$rmse[best_idx], 4), "\n")
  
  # Refit with best ncomp to get final model
  best_model <- pcr.rolling.window(Y, nprev, indice, ncomp = best_ncomp, 
                                   date_col = date_col)
  
  # Plot CV results
  par(mfrow = c(1, 2))
  plot(cv_results$ncomp, cv_results$rmse, type = "b", 
       xlab = "Number of Components", ylab = "RMSE", 
       main = "CV: RMSE vs ncomp")
  abline(v = best_ncomp, col = "blue", lty = 2)
  points(best_ncomp, cv_results$rmse[best_idx], col = "blue", pch = 19, cex = 1.5)
  
  plot(cv_results$ncomp, cv_results$mae, type = "b",
       xlab = "Number of Components", ylab = "MAE", 
       main = "CV: MAE vs ncomp")
  abline(v = best_ncomp, col = "blue", lty = 2)
  points(best_ncomp, cv_results$mae[best_idx], col = "blue", pch = 19, cex = 1.5)
  par(mfrow = c(1, 1))
  
  return(list(
    "best_ncomp" = best_ncomp,
    "best_model" = best_model,
    "cv_results" = cv_results
  ))
}


#' Example usage and comparison with LASSO:
#' 
#' # Assuming you have data matrix Y and date column dates
#' 
#' # PCR with CV to find optimal number of components
#' pcr_cv_results <- pcr.cv.ncomp(Y, nprev = 24, indice = 1, date_col = dates)
#' 
#' # PCR with fixed number of components
#' pcr_results <- pcr.rolling.window(Y, nprev = 24, indice = 1, ncomp = 3, date_col = dates)
#' 
#' # Compare with LASSO
#' lasso_cv_results <- lasso.cv.lambda(Y, nprev = 24, indice = 1, date_col = dates)
#' 
#' # Compare errors
#' cat("PCR RMSE:", pcr_cv_results$best_model$errors["rmse"], "\n")
#' cat("LASSO RMSE:", lasso_cv_results$best_model$errors["rmse"], "\n")