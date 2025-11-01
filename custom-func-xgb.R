# XGBoost Functions - Parallel to LASSO, PCR, PLS, and RF structure
# Tuning on nrounds (number of boosting rounds) parameter

# Required packages
# library(xgboost)  # For XGBoost - install with: install.packages("xgboost")

#' Function 1: Train XGBoost on first n-1 rows, predict last row with specified number of rounds
#'
#' @param Y Data matrix with all variables
#' @param indice Index of the response variable column
#' @param nrounds Number of boosting rounds
#' @return List containing model info and prediction
runxgb <- function(Y, indice, nrounds) {
  
  # Check if xgboost package is loaded
  if (!requireNamespace("xgboost", quietly = TRUE)) {
    stop("Package 'xgboost' is required. Please install it with: install.packages('xgboost')")
  }
  
  # Validate inputs
  if (nrow(Y) < 2) {
    stop("Y must have at least 2 rows")
  }
  if (indice < 1 || indice > ncol(Y)) {
    stop("indice must be a valid column index")
  }
  if (nrounds < 1) {
    stop("nrounds must be at least 1")
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
  
  # XGBoost requires numeric matrix (no column names needed but ensure numeric)
  X_train <- matrix(as.numeric(X_train), nrow = nrow(X_train), ncol = ncol(X_train))
  X_test <- matrix(as.numeric(X_test), nrow = nrow(X_test), ncol = ncol(X_test))
  
  # Create DMatrix objects (XGBoost's internal data structure)
  dtrain <- xgboost::xgb.DMatrix(data = X_train, label = y_train)
  dtest <- xgboost::xgb.DMatrix(data = X_test)
  
  # Set parameters for regression
  params <- list(
    objective = "reg:squarederror",  # Regression with squared error
    max_depth = 6,                   # Maximum tree depth (default)
    eta = 0.3,                       # Learning rate (default)
    subsample = 1,                   # Use all rows
    colsample_bytree = 1             # Use all columns
  )
  
  # Train XGBoost model
  # verbose = 0 to suppress training messages
  xgb_model <- xgboost::xgb.train(
    params = params,
    data = dtrain,
    nrounds = nrounds,
    verbose = 0
  )
  
  # Extract feature importance (gain)
  importance <- xgboost::xgb.importance(model = xgb_model)
  
  # Create a full importance vector for all features (some may have 0 importance)
  n_features <- ncol(X_train)
  full_importance <- rep(0, n_features)
  names(full_importance) <- paste0("X", 1:n_features)
  
  # Fill in the importance values (importance$Feature may not include all features)
  if (nrow(importance) > 0) {
    # Match feature names and assign gain values
    for (j in 1:nrow(importance)) {
      feat_idx <- as.numeric(gsub("f", "", importance$Feature[j])) + 1  # XGBoost uses 0-indexing
      full_importance[feat_idx] <- importance$Gain[j]
    }
  }
  
  # Predict
  pred <- predict(xgb_model, dtest)
  
  # Return results
  return(list(
    "xgb_model" = xgb_model,         # XGBoost model object
    "pred" = as.numeric(pred),       # Prediction
    "nrounds_used" = nrounds,        # Number of rounds actually used
    "importance" = full_importance   # Feature importance vector
  ))
}


#' Function 2: Rolling window forecasting with specified number of rounds (with date support)
#'
#' @param Y Data matrix with all variables
#' @param nprev Number of observations to test for
#' @param indice Index of the response variable column
#' @param nrounds Number of boosting rounds
#' @param date_col Optional date column for plotting
#' @return List containing predictions and errors
xgb.rolling.window <- function(Y, nprev, indice = 1, nrounds, date_col = NULL) {
  
  # Initialize storage
  n_features <- ncol(Y) - 1
  save.pred <- matrix(NA, nprev, 1)
  save.nrounds.used <- numeric(nprev)
  save.importance <- matrix(NA, nprev, n_features)  # Matrix to store importance over time
  colnames(save.importance) <- paste0("X", 1:n_features)
  
  # Rolling window loop
  for(i in nprev:1) {
    Y.window <- Y[(1 + nprev - i):(1 + nrow(Y) - i), ]
    xgb.model <- runxgb(Y.window, indice, nrounds = nrounds)
    
    # Save results
    save.pred[(1 + nprev - i), ] <- xgb.model$pred
    save.nrounds.used[(1 + nprev - i)] <- xgb.model$nrounds_used
    save.importance[(1 + nprev - i), ] <- xgb.model$importance
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
    plot(dates, real_test, type = "l", main = paste("XGBoost: nrounds =", nrounds),
         ylab = "Value", xlab = "Date", xaxt = "n")
    lines(dates, save.pred, col = "purple")
    
    # Show dates at regular intervals
    n_labels <- 10  # Adjust this number
    label_indices <- seq(1, length(dates), length.out = n_labels)
    axis.Date(1, at = dates[label_indices], format = "%Y-%m")
    
  } else {
    # Plot without dates (original behaviour)
    plot(real_test, type = "l", main = paste("XGBoost: nrounds =", nrounds),
         ylab = "Value", xlab = "Time Period")
    lines(save.pred, col = "purple")
  }
  
  legend("topright", legend = c("Actual", "Predicted"), 
         col = c("black", "purple"), lty = 1)
  
  # Compute errors
  rmse <- sqrt(mean((real_test - save.pred)^2))
  mae <- mean(abs(real_test - save.pred))
  errors <- c("rmse" = rmse, "mae" = mae)
  
  # Compute average importance across all windows
  avg_importance <- colMeans(save.importance)
  
  return(list(
    "pred" = save.pred, 
    "nrounds_used" = save.nrounds.used,
    "importance" = save.importance,      # Matrix: [time_periods, features]
    "avg_importance" = avg_importance,   # Average importance across windows
    "errors" = errors
  ))
}


#' Function 3: Cross-validation to find optimal number of boosting rounds
#'
#' @param Y Data matrix with all variables
#' @param nprev Number of observations to test for
#' @param indice Index of the response variable column
#' @param nrounds_grid Vector of round numbers to test (default: c(10, 50, 100))
#' @param date_col Optional date column for plotting
#' @return List containing best nrounds, best model, and CV results
xgb.cv.nrounds <- function(Y, nprev, indice = 1, 
                           nrounds_grid = c(10, 50, 100),
                           date_col = NULL) {
  
  # Store results for each nrounds
  cv_results <- data.frame(
    nrounds = nrounds_grid,
    rmse = NA,
    mae = NA
  )
  
  cat("Testing", length(nrounds_grid), "nrounds values...\n")
  
  # Test each nrounds value
  for(i in seq_along(nrounds_grid)) {
    nrounds_val <- nrounds_grid[i]
    
    # Run rolling window with this number of rounds
    result <- xgb.rolling.window(Y, nprev, indice, nrounds = nrounds_val, 
                                 date_col = date_col)
    
    # Store errors
    cv_results$rmse[i] <- result$errors["rmse"]
    cv_results$mae[i] <- result$errors["mae"]
    
    cat("Completed nrounds =", nrounds_val, "- RMSE:", 
        round(result$errors["rmse"], 4), "\n")
  }
  
  # Find best nrounds (minimum RMSE)
  best_idx <- which.min(cv_results$rmse)
  best_nrounds <- cv_results$nrounds[best_idx]
  
  cat("\nBest nrounds:", best_nrounds, "with RMSE:", 
      round(cv_results$rmse[best_idx], 4), "\n")
  
  # Refit with best nrounds to get final model
  best_model <- xgb.rolling.window(Y, nprev, indice, nrounds = best_nrounds, 
                                   date_col = date_col)
  
  # Plot CV results
  par(mfrow = c(1, 2))
  plot(cv_results$nrounds, cv_results$rmse, type = "b", 
       xlab = "Number of Rounds", ylab = "RMSE", 
       main = "CV: RMSE vs nrounds (XGBoost)")
  abline(v = best_nrounds, col = "purple", lty = 2)
  points(best_nrounds, cv_results$rmse[best_idx], col = "purple", pch = 19, cex = 1.5)
  
  plot(cv_results$nrounds, cv_results$mae, type = "b",
       xlab = "Number of Rounds", ylab = "MAE", 
       main = "CV: MAE vs nrounds (XGBoost)")
  abline(v = best_nrounds, col = "purple", lty = 2)
  points(best_nrounds, cv_results$mae[best_idx], col = "purple", pch = 19, cex = 1.5)
  par(mfrow = c(1, 1))
  
  return(list(
    "best_nrounds" = best_nrounds,
    "best_model" = best_model,
    "cv_results" = cv_results
  ))
}


#' Example usage and comparison with all methods:
#' 
#' # Assuming you have data matrix Y and date column dates
#' # Make sure to load the xgboost package first: library(xgboost)
#' 
#' # XGBoost with CV to find optimal number of rounds
#' xgb_cv_results <- xgb.cv.nrounds(Y, nprev = 24, indice = 1, date_col = dates)
#' 
#' # XGBoost with fixed number of rounds
#' xgb_results <- xgb.rolling.window(Y, nprev = 24, indice = 1, nrounds = 50, date_col = dates)
#' 
#' # Access feature importance
#' # Average importance across all windows
#' print(xgb_results$avg_importance)
#' 
#' # Importance matrix over time (each row is a time period)
#' head(xgb_results$importance)
#' 
#' # Plot feature importance for most important features
#' top_features <- order(xgb_results$avg_importance, decreasing = TRUE)[1:5]
#' barplot(xgb_results$avg_importance[top_features], 
#'         names.arg = names(xgb_results$avg_importance)[top_features],
#'         main = "Top 5 Most Important Features",
#'         ylab = "Average Gain",
#'         col = "purple")
#' 
#' # Track how importance of a specific feature changes over time
#' feature_to_track <- 1  # Track first feature
#' plot(xgb_results$importance[, feature_to_track], type = "l",
#'      main = paste("Importance of Feature", feature_to_track, "Over Time"),
#'      ylab = "Gain", xlab = "Time Period")
#' 
#' # Compare with RF
#' rf_cv_results <- rf.cv.ntree(Y, nprev = 24, indice = 1, date_col = dates)
#' 
#' # Compare with PLS
#' pls_cv_results <- pls.cv.ncomp(Y, nprev = 24, indice = 1, date_col = dates)
#' 
#' # Compare with PCR
#' pcr_cv_results <- pcr.cv.ncomp(Y, nprev = 24, indice = 1, date_col = dates)
#' 
#' # Compare with LASSO
#' lasso_cv_results <- lasso.cv.lambda(Y, nprev = 24, indice = 1, date_col = dates)
#' 
#' # Compare all five methods
#' cat("XGBoost RMSE:", xgb_cv_results$best_model$errors["rmse"], "\n")
#' cat("RF RMSE:", rf_cv_results$best_model$errors["rmse"], "\n")
#' cat("PLS RMSE:", pls_cv_results$best_model$errors["rmse"], "\n")
#' cat("PCR RMSE:", pcr_cv_results$best_model$errors["rmse"], "\n")
#' cat("LASSO RMSE:", lasso_cv_results$best_model$errors["rmse"], "\n")
#' 
#' # Expected behaviour:
#' # - XGBoost is gradient boosting (builds trees sequentially)
#' # - XGBoost often performs very well out-of-the-box
#' # - XGBoost can capture complex non-linear relationships
#' # - XGBoost is typically faster than RF
#' # - XGBoost is more sensitive to hyperparameter tuning than RF
#' # - Linear methods (LASSO/PLS/PCR) are faster and more interpretable
#' # - Feature importance shows which variables contribute most to predictions