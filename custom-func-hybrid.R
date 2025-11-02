# Hybrid EN + XGBoost Functions
# Sequential residual fitting: EN first, then XGB on residuals

# Required packages
# library(glmnet)   # For Elastic Net
# library(xgboost)  # For XGBoost

#' Function 1: Train Hybrid model on first n-1 rows, predict last row
#'
#' @param Y Data matrix with all variables
#' @param indice Index of the response variable column
#' @param lambda Regularisation parameter for Elastic Net
#' @param nrounds Number of boosting rounds for XGBoost
#' @param alpha Elastic Net mixing parameter (default 0.5)
#' @return List containing models, predictions, and feature info
runhybrid <- function(Y, indice, lambda, nrounds, alpha = 0.5) {
  
  # Check if required packages are loaded
  if (!requireNamespace("glmnet", quietly = TRUE)) {
    stop("Package 'glmnet' is required. Please install it with: install.packages('glmnet')")
  }
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
  
  # Extract training data (all rows except the last)
  y_train <- Y[1:(nrow(Y)-1), indice]
  X_train <- Y[1:(nrow(Y)-1), -indice, drop = FALSE]
  
  # Extract test data (only the last row)
  X_test <- Y[nrow(Y), -indice, drop = FALSE]
  
  # ===== STAGE 1: Elastic Net =====
  # Fit EN model
  en_model <- glmnet::glmnet(X_train, y_train, alpha = alpha, lambda = lambda)
  
  # Get EN predictions on training data
  en_pred_train <- predict(en_model, newx = X_train, s = lambda)
  en_pred_train <- as.numeric(en_pred_train)
  
  # Calculate residuals
  resid_train <- y_train - en_pred_train
  
  # Get EN prediction on test data
  en_pred_test <- predict(en_model, newx = X_test, s = lambda)
  en_pred_test <- as.numeric(en_pred_test)
  
  # ===== STAGE 2: XGBoost on Residuals =====
  # Prepare data for XGBoost
  X_train_xgb <- matrix(as.numeric(X_train), nrow = nrow(X_train), ncol = ncol(X_train))
  X_test_xgb <- matrix(as.numeric(X_test), nrow = nrow(X_test), ncol = ncol(X_test))
  
  # Create DMatrix objects
  dtrain <- xgboost::xgb.DMatrix(data = X_train_xgb, label = resid_train)
  dtest <- xgboost::xgb.DMatrix(data = X_test_xgb)
  
  # Set XGBoost parameters
  params <- list(
    objective = "reg:squarederror",
    max_depth = 6,
    eta = 0.3,
    subsample = .8,
    colsample_bytree = .8
  )
  
  # Train XGBoost model on residuals
  xgb_model <- xgboost::xgb.train(
    params = params,
    data = dtrain,
    nrounds = nrounds,
    verbose = 0
  )
  
  # Get XGBoost prediction on test data (predicting the residual)
  xgb_pred_test <- predict(xgb_model, dtest)
  xgb_pred_test <- as.numeric(xgb_pred_test)
  
  # ===== FINAL HYBRID PREDICTION =====
  final_pred <- en_pred_test + xgb_pred_test
  
  # ===== Extract feature information =====
  # EN coefficients
  en_coef <- as.numeric(coef(en_model, s = lambda))
  
  # XGBoost importance
  importance <- xgboost::xgb.importance(model = xgb_model)
  n_features <- ncol(X_train)
  full_importance <- rep(0, n_features)
  names(full_importance) <- paste0("X", 1:n_features)
  
  if (nrow(importance) > 0) {
    for (j in 1:nrow(importance)) {
      feat_idx <- as.numeric(gsub("f", "", importance$Feature[j])) + 1
      full_importance[feat_idx] <- importance$Gain[j]
    }
  }
  
  # Return everything
  return(list(
    "en_model" = en_model,
    "xgb_model" = xgb_model,
    "final_pred" = final_pred,
    "en_pred" = en_pred_test,
    "xgb_pred" = xgb_pred_test,
    "en_coef" = en_coef,
    "xgb_importance" = full_importance
  ))
}


#' Function 2: Rolling window forecasting with Hybrid model (with date support)
#'
#' @param Y Data matrix with all variables
#' @param nprev Number of observations to test for
#' @param indice Index of the response variable column
#' @param lambda Regularisation parameter for Elastic Net
#' @param nrounds Number of boosting rounds for XGBoost
#' @param alpha Elastic Net mixing parameter (default 0.5)
#' @param date_col Optional date column for plotting
#' @return List containing predictions, errors, and feature information
hybrid.rolling.window <- function(Y, nprev, indice = 1, lambda, nrounds, 
                                  alpha = 0.5, date_col = NULL) {
  
  # Initialize storage
  n_features <- ncol(Y)
  save.pred <- matrix(NA, nprev, 1)
  save.en.pred <- matrix(NA, nprev, 1)
  save.xgb.pred <- matrix(NA, nprev, 1)
  save.en.coef <- matrix(NA, nprev, n_features)
  save.xgb.importance <- matrix(NA, nprev, n_features - 1)
  colnames(save.xgb.importance) <- paste0("X", 1:(n_features - 1))
  
  # Rolling window loop
  for(i in nprev:1) {
    Y.window <- Y[(1 + nprev - i):(1 + nrow(Y) - i), ]
    hybrid.model <- runhybrid(Y.window, indice, lambda = lambda, 
                              nrounds = nrounds, alpha = alpha)
    
    # Save results
    save.pred[(1 + nprev - i), ] <- hybrid.model$final_pred
    save.en.pred[(1 + nprev - i), ] <- hybrid.model$en_pred
    save.xgb.pred[(1 + nprev - i), ] <- hybrid.model$xgb_pred
    save.en.coef[(1 + nprev - i), ] <- hybrid.model$en_coef
    save.xgb.importance[(1 + nprev - i), ] <- hybrid.model$xgb_importance
  }
  
  # Get actual values
  real <- Y[, indice]
  real_test <- tail(real, nprev)
  
  # Handle dates if provided
  if (!is.null(date_col)) {
    dates <- tail(date_col, nprev)
    
    # Convert YYYYMM to Date format for better plotting
    if (is.numeric(dates)) {
      year <- floor(dates / 100)
      month <- dates %% 100
      dates <- as.Date(paste(year, month, "01", sep = "-"))
    } else if (!inherits(dates, "Date")) {
      dates <- as.Date(as.character(dates), format = "%Y%m")
    }
    
    # Plot with dates on x-axis
    plot(dates, real_test, type = "l", 
         main = paste("Hybrid (EN + XGB): λ =", round(lambda, 6), ", nrounds =", nrounds),
         ylab = "Value", xlab = "Date", xaxt = "n", lwd = 2)
    lines(dates, save.pred, col = "blue", lwd = 2)
    
    # Show dates at regular intervals
    n_labels <- 10
    label_indices <- seq(1, length(dates), length.out = n_labels)
    axis.Date(1, at = dates[label_indices], format = "%Y-%m")
    
  } else {
    # Plot without dates
    plot(real_test, type = "l", 
         main = paste("Hybrid (EN + XGB): λ =", round(lambda, 6), ", nrounds =", nrounds),
         ylab = "Value", xlab = "Time Period", lwd = 2)
    lines(save.pred, col = "blue", lwd = 2)
  }
  
  legend("topright", legend = c("Actual", "Hybrid Predicted"), 
         col = c("black", "blue"), lty = 1, lwd = 2)
  
  # Compute errors
  rmse <- sqrt(mean((real_test - save.pred)^2))
  mae <- mean(abs(real_test - save.pred))
  errors <- c("rmse" = rmse, "mae" = mae)
  
  # Compute average importance across all windows
  avg_xgb_importance <- colMeans(save.xgb.importance)
  
  return(list(
    "pred" = save.pred,
    "en_pred" = save.en.pred,
    "xgb_pred" = save.xgb.pred,
    "en_coef" = save.en.coef,
    "xgb_importance" = save.xgb.importance,
    "avg_xgb_importance" = avg_xgb_importance,
    "errors" = errors
  ))
}


#' Function 3: Cross-validation to find optimal lambda and nrounds simultaneously
#'
#' @param Y Data matrix with all variables
#' @param nprev Number of observations to test for
#' @param indice Index of the response variable column
#' @param alpha Elastic Net mixing parameter (default 0.5)
#' @param lambda_grid Vector of lambda values to test (default: c(0.001, 0.01, 0.1))
#' @param nrounds_grid Vector of nrounds values to test (default: c(50, 100, 150))
#' @param date_col Optional date column for plotting
#' @return List containing best parameters, best model, and CV results
hybrid.cv.params <- function(Y, nprev, indice = 1, alpha = 0.5,
                             lambda_grid = c(0.001, 0.01, 0.1),
                             nrounds_grid = c(50, 100, 150),
                             date_col = NULL) {
  
  # Create full grid of parameter combinations
  param_grid <- expand.grid(
    lambda = lambda_grid,
    nrounds = nrounds_grid
  )
  
  # Store results for each combination
  cv_results <- data.frame(
    lambda = param_grid$lambda,
    nrounds = param_grid$nrounds,
    rmse = NA,
    mae = NA,
    xgb_contribution = NA  # Will calculate RMSE improvement over EN-only
  )
  
  # Store EN-only performance for comparison
  en_only_rmse <- NA
  
  cat("Testing", nrow(param_grid), "parameter combinations (", 
      length(lambda_grid), "lambdas ×", length(nrounds_grid), "nrounds)...\n\n")
  
  # Test each parameter combination
  for(i in 1:nrow(param_grid)) {
    lambda_val <- param_grid$lambda[i]
    nrounds_val <- param_grid$nrounds[i]
    
    cat("Testing combination", i, "of", nrow(param_grid), 
        "- Lambda:", lambda_val, ", nrounds:", nrounds_val, "\n")
    
    # Run rolling window with this parameter combination
    result <- hybrid.rolling.window(Y, nprev, indice, 
                                    lambda = lambda_val, 
                                    nrounds = nrounds_val,
                                    alpha = alpha, 
                                    date_col = date_col)
    
    # Store errors
    cv_results$rmse[i] <- result$errors["rmse"]
    cv_results$mae[i] <- result$errors["mae"]
    
    # Calculate EN-only performance (using just the EN component)
    real <- Y[, indice]
    real_test <- tail(real, nprev)
    en_only_rmse_current <- sqrt(mean((real_test - result$en_pred)^2))
    
    # Store EN-only RMSE from first iteration for reference
    if (i == 1) {
      en_only_rmse <- en_only_rmse_current
    }
    
    # Calculate XGBoost contribution (negative means XGB made it worse)
    cv_results$xgb_contribution[i] <- en_only_rmse_current - cv_results$rmse[i]
    
    cat("  → Hybrid RMSE:", round(cv_results$rmse[i], 4), 
        ", EN-only RMSE:", round(en_only_rmse_current, 4),
        ", XGB contribution:", round(cv_results$xgb_contribution[i], 4), "\n\n")
  }
  
  # Find best combination (minimum RMSE)
  best_idx <- which.min(cv_results$rmse)
  best_lambda <- cv_results$lambda[best_idx]
  best_nrounds <- cv_results$nrounds[best_idx]
  
  cat("=" , rep("=", 70), "\n", sep = "")
  cat("BEST PARAMETERS:\n")
  cat("Lambda:", best_lambda, ", nrounds:", best_nrounds, "\n")
  cat("Hybrid RMSE:", round(cv_results$rmse[best_idx], 4), "\n")
  cat("XGB Contribution:", round(cv_results$xgb_contribution[best_idx], 4), "\n")
  cat("=" , rep("=", 70), "\n\n", sep = "")
  
  # Refit with best parameters to get final model
  cat("Refitting with best parameters...\n")
  best_model <- hybrid.rolling.window(Y, nprev, indice, 
                                      lambda = best_lambda, 
                                      nrounds = best_nrounds,
                                      alpha = alpha, 
                                      date_col = date_col)
  
  # Create visualisations
  par(mfrow = c(2, 2))
  
  # Plot 1: RMSE heatmap
  rmse_matrix <- matrix(cv_results$rmse, 
                        nrow = length(lambda_grid), 
                        ncol = length(nrounds_grid))
  
  image(1:length(lambda_grid), 1:length(nrounds_grid), rmse_matrix,
        xlab = "Lambda Index", ylab = "nrounds Index",
        main = "RMSE Heatmap (Hybrid)",
        col = heat.colors(20, rev = TRUE),
        xaxt = "n", yaxt = "n")
  axis(1, at = 1:length(lambda_grid), labels = lambda_grid)
  axis(2, at = 1:length(nrounds_grid), labels = nrounds_grid)
  points(which(lambda_grid == best_lambda), 
         which(nrounds_grid == best_nrounds),
         pch = 19, cex = 2, col = "blue")
  
  # Plot 2: MAE heatmap
  mae_matrix <- matrix(cv_results$mae, 
                       nrow = length(lambda_grid), 
                       ncol = length(nrounds_grid))
  
  image(1:length(lambda_grid), 1:length(nrounds_grid), mae_matrix,
        xlab = "Lambda Index", ylab = "nrounds Index",
        main = "MAE Heatmap (Hybrid)",
        col = heat.colors(20, rev = TRUE),
        xaxt = "n", yaxt = "n")
  axis(1, at = 1:length(lambda_grid), labels = lambda_grid)
  axis(2, at = 1:length(nrounds_grid), labels = nrounds_grid)
  points(which(lambda_grid == best_lambda), 
         which(nrounds_grid == best_nrounds),
         pch = 19, cex = 2, col = "blue")
  
  # Plot 3: XGB Contribution
  contrib_matrix <- matrix(cv_results$xgb_contribution,
                           nrow = length(lambda_grid),
                           ncol = length(nrounds_grid))
  
  image(1:length(lambda_grid), 1:length(nrounds_grid), contrib_matrix,
        xlab = "Lambda Index", ylab = "nrounds Index",
        main = "XGB Contribution (Positive = Improvement)",
        col = cm.colors(20),
        xaxt = "n", yaxt = "n")
  axis(1, at = 1:length(lambda_grid), labels = lambda_grid)
  axis(2, at = 1:length(nrounds_grid), labels = nrounds_grid)
  points(which(lambda_grid == best_lambda), 
         which(nrounds_grid == best_nrounds),
         pch = 19, cex = 2, col = "blue")
  abline(h = 0, lty = 2)
  
  # Plot 4: RMSE vs combination index
  plot(1:nrow(cv_results), cv_results$rmse, type = "b",
       xlab = "Parameter Combination Index", ylab = "RMSE",
       main = "RMSE Across All Combinations",
       pch = 19, col = "darkgrey")
  points(best_idx, cv_results$rmse[best_idx], 
         col = "blue", pch = 19, cex = 2)
  abline(h = cv_results$rmse[best_idx], col = "blue", lty = 2)
  
  par(mfrow = c(1, 1))
  
  return(list(
    "best_lambda" = best_lambda,
    "best_nrounds" = best_nrounds,
    "best_model" = best_model,
    "cv_results" = cv_results,
    "en_only_rmse" = en_only_rmse
  ))
}


#' Example usage:
#' 
#' # Load required packages
#' library(glmnet)
#' library(xgboost)
#' 
#' # Assuming you have data matrix Y and date column dates
#' 
#' # Hybrid with CV to find optimal parameters
#' hybrid_cv_results <- hybrid.cv.params(Y, nprev = 24, indice = 1, date_col = dates)
#' 
#' # Hybrid with fixed parameters
#' hybrid_results <- hybrid.rolling.window(Y, nprev = 24, indice = 1, 
#'                                         lambda = 0.01, nrounds = 100, 
#'                                         date_col = dates)
#' 
#' # Access separate components
#' # EN predictions only
#' head(hybrid_results$en_pred)
#' 
#' # XGB predictions only (residual predictions)
#' head(hybrid_results$xgb_pred)
#' 
#' # Final hybrid predictions (EN + XGB)
#' head(hybrid_results$pred)
#' 
#' # EN coefficients over time
#' head(hybrid_results$en_coef)
#' 
#' # XGB feature importance
#' print(hybrid_results$avg_xgb_importance)
#' 
#' # Compare with individual methods
#' lasso_cv_results <- lasso.cv.lambda(Y, nprev = 24, indice = 1, date_col = dates)
#' xgb_cv_results <- xgb.cv.nrounds(Y, nprev = 24, indice = 1, date_col = dates)
#' 
#' cat("Hybrid RMSE:", hybrid_cv_results$best_model$errors["rmse"], "\n")
#' cat("LASSO RMSE:", lasso_cv_results$best_model$errors["rmse"], "\n")
#' cat("XGBoost RMSE:", xgb_cv_results$best_model$errors["rmse"], "\n")
#' cat("XGB Contribution to Hybrid:", 
#'     hybrid_cv_results$cv_results$xgb_contribution[
#'       which.min(hybrid_cv_results$cv_results$rmse)], "\n")
#' 
#' # Visualise contribution of each component
#' par(mfrow = c(1, 2))
#' 
#' # Plot EN component
#' plot(hybrid_results$en_pred, type = "l", col = "red", 
#'      main = "EN Component", ylab = "Prediction")
#' 
#' # Plot XGB component (residuals)
#' plot(hybrid_results$xgb_pred, type = "l", col = "purple",
#'      main = "XGB Component (Residuals)", ylab = "Residual Prediction")
#' 
#' par(mfrow = c(1, 1))
#' 
#' # Expected behaviour:
#' # - Hybrid should potentially outperform either method alone
#' # - EN captures linear relationships
#' # - XGB captures non-linear patterns in the residuals
#' # - XGB contribution shows whether the 2-stage approach adds value
#' # - If XGB contribution is negative, EN alone might be sufficient