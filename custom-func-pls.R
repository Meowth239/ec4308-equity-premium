# PLS Functions - Parallel to LASSO and PCR structure
# Using pls package with plsr() function

# Required packages
# library(glmnet)  # For LASSO (already in your code)
# library(pls)     # For PLS - install with: install.packages("pls")

#' Function 1: Train PLS on first n-1 rows, predict last row with specified number of components
#'
#' @param Y Data matrix with all variables
#' @param indice Index of the response variable column
#' @param ncomp Number of PLS components to use
#' @return List containing model info and prediction
runpls <- function(Y, indice, ncomp) {
  
  # Check if pls package is loaded
  if (!requireNamespace("pls", quietly = TRUE)) {
    stop("Package 'pls' is required. Please install it with: install.packages('pls')")
  }
  
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
  
  # Ensure Y is a matrix
  if (!is.matrix(Y)) {
    Y <- as.matrix(Y)
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
  
  # Create safe column names (avoid special characters)
  safe_names <- paste0("X", 1:ncol(X_train))
  colnames(X_train) <- safe_names
  colnames(X_test) <- safe_names
  
  # Create data frame for training
  train_df <- as.data.frame(X_train)
  train_df$Y <- y_train
  
  # Fit PLS model using formula interface with safe names
  pls_model <- pls::plsr(Y ~ ., data = train_df, 
                         ncomp = ncomp, 
                         scale = TRUE,
                         validation = "none")
  
  # Create test data frame with same structure
  test_df <- as.data.frame(X_test)
  
  # Predict - extract the prediction for the specified ncomp
  pred_result <- predict(pls_model, newdata = test_df, ncomp = ncomp)
  
  # Convert to numeric - pred_result should be array-like
  pred <- as.numeric(pred_result[1, 1, 1])
  
  # Return results
  return(list(
    "pls_model" = pls_model,         # PLS model object
    "pred" = pred,                   # Prediction
    "ncomp_used" = ncomp             # Number of components actually used
  ))
}


#' Function 2: Rolling window forecasting with specified number of components (with date support)
#'
#' @param Y Data matrix with all variables
#' @param nprev Number of observations to test for
#' @param indice Index of the response variable column
#' @param ncomp Number of PLS components to use
#' @param date_col Optional date column for plotting
#' @return List containing predictions and errors
pls.rolling.window <- function(Y, nprev, indice = 1, ncomp, date_col = NULL) {
  
  # Initialize storage
  save.pred <- matrix(NA, nprev, 1)
  save.ncomp.used <- numeric(nprev)
  
  # Rolling window loop
  for(i in nprev:1) {
    Y.window <- Y[(1 + nprev - i):(1 + nrow(Y) - i), ]
    pls.model <- runpls(Y.window, indice, ncomp = ncomp)
    
    # Save results
    save.pred[(1 + nprev - i), ] <- pls.model$pred
    save.ncomp.used[(1 + nprev - i)] <- pls.model$ncomp_used
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
    plot(dates, real_test, type = "l", main = paste("PLS: ncomp =", ncomp),
         ylab = "Value", xlab = "Date", xaxt = "n")
    lines(dates, save.pred, col = "red")
    
    # Show dates at regular intervals
    n_labels <- 10  # Adjust this number
    label_indices <- seq(1, length(dates), length.out = n_labels)
    axis.Date(1, at = dates[label_indices], format = "%Y-%m")
    
  } else {
    # Plot without dates (original behaviour)
    plot(real_test, type = "l", main = paste("PLS: ncomp =", ncomp),
         ylab = "Value", xlab = "Time Period")
    lines(save.pred, col = "red")
  }
  
  legend("topright", legend = c("Actual", "Predicted"), 
         col = c("black", "red"), lty = 1)
  
  # Compute errors
  rmse <- sqrt(mean((real_test - save.pred)^2))
  mae <- mean(abs(real_test - save.pred))
  errors <- c("rmse" = rmse, "mae" = mae)
  
  return(list(
    "pred" = save.pred, 
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
#' @note PLS uses y in component construction, so may overfit more easily than PCR with small nprev
pls.cv.ncomp <- function(Y, nprev, indice = 1, 
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
  cat("Note: PLS uses y in component construction and may overfit with small samples.\n\n")
  
  # Test each ncomp value
  for(i in seq_along(ncomp_grid)) {
    ncomp_val <- ncomp_grid[i]
    
    # Run rolling window with this number of components
    result <- pls.rolling.window(Y, nprev, indice, ncomp = ncomp_val, 
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
  best_model <- pls.rolling.window(Y, nprev, indice, ncomp = best_ncomp, 
                                   date_col = date_col)
  
  # Plot CV results
  par(mfrow = c(1, 2))
  plot(cv_results$ncomp, cv_results$rmse, type = "b", 
       xlab = "Number of Components", ylab = "RMSE", 
       main = "CV: RMSE vs ncomp (PLS)")
  abline(v = best_ncomp, col = "red", lty = 2)
  points(best_ncomp, cv_results$rmse[best_idx], col = "red", pch = 19, cex = 1.5)
  
  plot(cv_results$ncomp, cv_results$mae, type = "b",
       xlab = "Number of Components", ylab = "MAE", 
       main = "CV: MAE vs ncomp (PLS)")
  abline(v = best_ncomp, col = "red", lty = 2)
  points(best_ncomp, cv_results$mae[best_idx], col = "red", pch = 19, cex = 1.5)
  par(mfrow = c(1, 1))
  
  return(list(
    "best_ncomp" = best_ncomp,
    "best_model" = best_model,
    "cv_results" = cv_results
  ))
}


#' Example usage and comparison with LASSO, PCR, and PLS:
#' 
#' # Assuming you have data matrix Y and date column dates
#' # Make sure to load the pls package first: library(pls)
#' 
#' # PLS with CV to find optimal number of components
#' pls_cv_results <- pls.cv.ncomp(Y, nprev = 24, indice = 1, date_col = dates)
#' 
#' # PLS with fixed number of components
#' pls_results <- pls.rolling.window(Y, nprev = 24, indice = 1, ncomp = 3, date_col = dates)
#' 
#' # Compare with PCR
#' pcr_cv_results <- pcr.cv.ncomp(Y, nprev = 24, indice = 1, date_col = dates)
#' 
#' # Compare with LASSO
#' lasso_cv_results <- lasso.cv.lambda(Y, nprev = 24, indice = 1, date_col = dates)
#' 
#' # Compare all three methods
#' cat("PLS RMSE:", pls_cv_results$best_model$errors["rmse"], "\n")
#' cat("PCR RMSE:", pcr_cv_results$best_model$errors["rmse"], "\n")
#' cat("LASSO RMSE:", lasso_cv_results$best_model$errors["rmse"], "\n")
#' 
#' # Expected behaviour:
#' # - PLS often performs better than PCR (uses y information)
#' # - PLS may overfit with small samples or high-dimensional data
#' # - LASSO provides sparse solutions (feature selection)
#' # - PCR/PLS provide dense solutions (all features contribute)