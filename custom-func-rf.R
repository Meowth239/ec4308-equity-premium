# Random Forest Functions - Parallel to LASSO, PCR, PLS, and XGBoost structure
# Tuning on ntree (number of trees) parameter

# Required packages
# library(randomForest)  # For RF - install with: install.packages("randomForest")

#' Function 1: Train RF on first n-1 rows, predict last row with specified number of trees
#'
#' @param Y Data matrix with all variables
#' @param indice Index of the response variable column
#' @param ntree Number of trees to grow
#' @return List containing model info and prediction
runrf <- function(Y, indice, ntree) {
  
  # Check if randomForest package is loaded
  if (!requireNamespace("randomForest", quietly = TRUE)) {
    stop("Package 'randomForest' is required. Please install it with: install.packages('randomForest')")
  }
  
  # Validate inputs
  if (nrow(Y) < 2) {
    stop("Y must have at least 2 rows")
  }
  if (indice < 1 || indice > ncol(Y)) {
    stop("indice must be a valid column index")
  }
  if (ntree < 1) {
    stop("ntree must be at least 1")
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
  
  # Create safe column names (avoid special characters)
  safe_names <- paste0("X", 1:ncol(X_train))
  colnames(X_train) <- safe_names
  colnames(X_test) <- safe_names
  
  # Create data frame for training
  train_df <- as.data.frame(X_train)
  train_df$Y <- y_train
  
  # Fit Random Forest model with importance = TRUE
  # mtry is automatically set to floor(sqrt(p)) for regression
  # nodesize defaults to 5 for regression
  rf_model <- randomForest::randomForest(Y ~ ., data = train_df, 
                                         ntree = ntree,
                                         importance = TRUE)
  
  # Create test data frame
  test_df <- as.data.frame(X_test)
  
  # Predict
  pred <- predict(rf_model, newdata = test_df)
  
  # Extract feature importance (IncNodePurity)
  raw_importance <- randomForest::importance(rf_model, type = 2)  # type = 2 for IncNodePurity
  
  # Normalise importance scores to sum to 1
  importance_vec <- as.vector(raw_importance)
  normalised_importance <- importance_vec / sum(importance_vec)
  names(normalised_importance) <- safe_names
  
  # Return results
  return(list(
    "rf_model" = rf_model,               # RF model object
    "pred" = as.numeric(pred),           # Prediction
    "ntree_used" = ntree,                # Number of trees actually used
    "importance" = normalised_importance # Normalised feature importance vector
  ))
}


#' Function 2: Rolling window forecasting with specified number of trees (with date support)
#'
#' @param Y Data matrix with all variables
#' @param nprev Number of observations to test for
#' @param indice Index of the response variable column
#' @param ntree Number of trees to grow
#' @param date_col Optional date column for plotting
#' @return List containing predictions and errors
rf.rolling.window <- function(Y, nprev, indice = 1, ntree, date_col = NULL) {
  
  # Initialize storage
  n_features <- ncol(Y) - 1
  save.pred <- matrix(NA, nprev, 1)
  save.ntree.used <- numeric(nprev)
  save.importance <- matrix(NA, nprev, n_features)  # Matrix to store importance over time
  colnames(save.importance) <- paste0("X", 1:n_features)
  
  # Rolling window loop
  for(i in nprev:1) {
    Y.window <- Y[(1 + nprev - i):(1 + nrow(Y) - i), ]
    rf.model <- runrf(Y.window, indice, ntree = ntree)
    
    # Save results
    save.pred[(1 + nprev - i), ] <- rf.model$pred
    save.ntree.used[(1 + nprev - i)] <- rf.model$ntree_used
    save.importance[(1 + nprev - i), ] <- rf.model$importance
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
    plot(dates, real_test, type = "l", main = paste("RF: ntree =", ntree),
         ylab = "Value", xlab = "Date", xaxt = "n")
    lines(dates, save.pred, col = "orange")
    
    # Show dates at regular intervals
    n_labels <- 10  # Adjust this number
    label_indices <- seq(1, length(dates), length.out = n_labels)
    axis.Date(1, at = dates[label_indices], format = "%Y-%m")
    
  } else {
    # Plot without dates (original behaviour)
    plot(real_test, type = "l", main = paste("RF: ntree =", ntree),
         ylab = "Value", xlab = "Time Period")
    lines(save.pred, col = "orange")
  }
  
  legend("topright", legend = c("Actual", "Predicted"), 
         col = c("black", "orange"), lty = 1)
  
  # Compute errors
  rmse <- sqrt(mean((real_test - save.pred)^2))
  mae <- mean(abs(real_test - save.pred))
  errors <- c("rmse" = rmse, "mae" = mae)
  
  # Compute average importance across all windows
  avg_importance <- colMeans(save.importance)
  
  return(list(
    "pred" = save.pred, 
    "ntree_used" = save.ntree.used,
    "importance" = save.importance,      # Matrix: [time_periods, features]
    "avg_importance" = avg_importance,   # Average importance across windows
    "errors" = errors
  ))
}


#' Function 3: Cross-validation to find optimal number of trees
#'
#' @param Y Data matrix with all variables
#' @param nprev Number of observations to test for
#' @param indice Index of the response variable column
#' @param ntree_grid Vector of tree numbers to test (default: c(10, 50, 100, 200))
#' @param date_col Optional date column for plotting
#' @return List containing best ntree, best model, and CV results
rf.cv.ntree <- function(Y, nprev, indice = 1, 
                        ntree_grid = c(10, 50, 100, 200),
                        date_col = NULL) {
  
  # Store results for each ntree
  cv_results <- data.frame(
    ntree = ntree_grid,
    rmse = NA,
    mae = NA
  )
  
  cat("Testing", length(ntree_grid), "ntree values...\n")
  
  # Test each ntree value
  for(i in seq_along(ntree_grid)) {
    ntree_val <- ntree_grid[i]
    
    # Run rolling window with this number of trees
    result <- rf.rolling.window(Y, nprev, indice, ntree = ntree_val, 
                                date_col = date_col)
    
    # Store errors
    cv_results$rmse[i] <- result$errors["rmse"]
    cv_results$mae[i] <- result$errors["mae"]
    
    cat("Completed ntree =", ntree_val, "- RMSE:", 
        round(result$errors["rmse"], 4), "\n")
  }
  
  # Find best ntree (minimum RMSE)
  best_idx <- which.min(cv_results$rmse)
  best_ntree <- cv_results$ntree[best_idx]
  
  cat("\nBest ntree:", best_ntree, "with RMSE:", 
      round(cv_results$rmse[best_idx], 4), "\n")
  
  # Refit with best ntree to get final model
  best_model <- rf.rolling.window(Y, nprev, indice, ntree = best_ntree, 
                                  date_col = date_col)
  
  # Plot CV results
  par(mfrow = c(1, 2))
  plot(cv_results$ntree, cv_results$rmse, type = "b", 
       xlab = "Number of Trees", ylab = "RMSE", 
       main = "CV: RMSE vs ntree (RF)")
  abline(v = best_ntree, col = "orange", lty = 2)
  points(best_ntree, cv_results$rmse[best_idx], col = "orange", pch = 19, cex = 1.5)
  
  plot(cv_results$ntree, cv_results$mae, type = "b",
       xlab = "Number of Trees", ylab = "MAE", 
       main = "CV: MAE vs ntree (RF)")
  abline(v = best_ntree, col = "orange", lty = 2)
  points(best_ntree, cv_results$mae[best_idx], col = "orange", pch = 19, cex = 1.5)
  par(mfrow = c(1, 1))
  
  return(list(
    "best_ntree" = best_ntree,
    "best_model" = best_model,
    "cv_results" = cv_results
  ))
}


#' Example usage and comparison with all methods:
#' 
#' # Assuming you have data matrix Y and date column dates
#' # Make sure to load the randomForest package first: library(randomForest)
#' 
#' # RF with CV to find optimal number of trees
#' rf_cv_results <- rf.cv.ntree(Y, nprev = 24, indice = 1, date_col = dates)
#' 
#' # RF with fixed number of trees
#' rf_results <- rf.rolling.window(Y, nprev = 24, indice = 1, ntree = 100, date_col = dates)
#' 
#' # Access feature importance
#' # Average importance across all windows
#' print(rf_results$avg_importance)
#' 
#' # Importance matrix over time (each row is a time period)
#' head(rf_results$importance)
#' 
#' # Plot feature importance for most important features
#' top_features <- order(rf_results$avg_importance, decreasing = TRUE)[1:5]
#' barplot(rf_results$avg_importance[top_features], 
#'         names.arg = names(rf_results$avg_importance)[top_features],
#'         main = "Top 5 Most Important Features (RF)",
#'         ylab = "Normalised Importance (IncNodePurity)",
#'         col = "orange")
#' 
#' # Track how importance of a specific feature changes over time
#' feature_to_track <- 1  # Track first feature
#' plot(rf_results$importance[, feature_to_track], type = "l",
#'      main = paste("Importance of Feature", feature_to_track, "Over Time (RF)"),
#'      ylab = "Normalised Importance", xlab = "Time Period",
#'      col = "orange")
#' 
#' # Compare with XGBoost
#' xgb_cv_results <- xgb.cv.nrounds(Y, nprev = 24, indice = 1, date_col = dates)
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
#' cat("RF RMSE:", rf_cv_results$best_model$errors["rmse"], "\n")
#' cat("XGBoost RMSE:", xgb_cv_results$best_model$errors["rmse"], "\n")
#' cat("PLS RMSE:", pls_cv_results$best_model$errors["rmse"], "\n")
#' cat("PCR RMSE:", pcr_cv_results$best_model$errors["rmse"], "\n")
#' cat("LASSO RMSE:", lasso_cv_results$best_model$errors["rmse"], "\n")
#' 
#' # Compare feature importance between RF and XGBoost
#' par(mfrow = c(1, 2))
#' barplot(rf_results$avg_importance, main = "RF Feature Importance",
#'         ylab = "Normalised Importance", col = "orange")
#' barplot(xgb_results$avg_importance, main = "XGBoost Feature Importance",
#'         ylab = "Average Gain", col = "purple")
#' par(mfrow = c(1, 1))
#' 
#' # Expected behaviour:
#' # - RF is non-parametric and can capture non-linear relationships
#' # - RF is robust to outliers and doesn't require feature scaling
#' # - RF may perform better with complex, non-linear data
#' # - RF feature importance (IncNodePurity) measures total reduction in node impurity
#' # - XGBoost feature importance (Gain) measures average gain across all splits
#' # - Both importance measures are now normalised for easier comparison
#' # - LASSO provides sparse, interpretable solutions
#' # - PLS/PCR provide linear dimension reduction
#' # - RF typically takes longer to train than linear methods but faster than heavily-tuned XGBoost</document_content></document>