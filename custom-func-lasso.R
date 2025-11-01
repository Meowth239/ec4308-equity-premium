#Adapted from Medeiros

#Y is the data
#indice is the index for pred var
#lambda is regularisation param
#alpha is determine Lasso(alpha = 1), Ridge(alpha = 0), EN(alpha = 0.5)

# Function 1: Train on first n-1 rows, predict last row with specified lambda
runlasso <- function(Y, indice, lambda, alpha = 1) {
  
  # Validate inputs
  if (nrow(Y) < 2) {
    stop("Y must have at least 2 rows")
  }
  if (indice < 1 || indice > ncol(Y)) {
    stop("indice must be a valid column index")
  }
  
  # Extract training data (all rows except the last)
  y_train <- Y[1:(nrow(Y)-1), indice]
  X_train <- Y[1:(nrow(Y)-1), -indice, drop = FALSE]
  
  # Extract test data (only the last row)
  X_test <- Y[nrow(Y), -indice, drop = FALSE]
  
  # Fit glmnet model with specified lambda
  model <- glmnet(X_train, y_train, alpha = alpha, lambda = lambda)
  
  # Generate prediction
  pred <- predict(model, newx = X_test, s = lambda)
  
  return(list("model" = model, "pred" = as.numeric(pred)))
}

#Y is the data
#nprev is number of obs to test for
#indice is the index for pred var
#lambda is regularisation param
#alpha is determine Lasso(alpha = 1), Ridge(alpha = 0), EN(alpha = 0.5)
#datecol to plot dates

# Function 2: Rolling window forecasting with specified lambda (with date support)
lasso.rolling.window <- function(Y, nprev, indice = 1, lambda, alpha = 1, date_col = NULL) {
  
  save.coef <- matrix(NA, nprev, ncol(Y))
  save.pred <- matrix(NA, nprev, 1)
  
  for(i in nprev:1) {
    Y.window <- Y[(1 + nprev - i):(1 + nrow(Y) - i), ]
    lasso.model <- runlasso(Y.window, indice, lambda = lambda, alpha = alpha)
    save.coef[(1 + nprev - i), ] <- as.numeric(coef(lasso.model$model, s = lambda))
    save.pred[(1 + nprev - i), ] <- lasso.model$pred
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
    plot(dates, real_test, type = "l", main = paste("Lambda =", round(lambda, 6)),
         ylab = "Value", xlab = "Date", xaxt = "n")
    lines(dates, save.pred, col = "red")
    
    # Show dates at regular intervals
    n_labels <- 10  # Adjust this number
    label_indices <- seq(1, length(dates), length.out = n_labels)
    axis.Date(1, at = dates[label_indices], format = "%Y-%m")
    
  } else {
    # Plot without dates (original behaviour)
    plot(real_test, type = "l", main = paste("Lambda =", round(lambda, 6)),
         ylab = "Value", xlab = "Time Period")
    lines(save.pred, col = "red")
  }
  
  legend("topright", legend = c("Actual", "Predicted"), 
         col = c("black", "red"), lty = 1)
  
  # Compute errors
  rmse <- sqrt(mean((real_test - save.pred)^2))
  mae <- mean(abs(real_test - save.pred))
  errors <- c("rmse" = rmse, "mae" = mae)
  
  return(list("pred" = save.pred, "coef" = save.coef, "errors" = errors))
}

#Y is the data
#nprev is number of obs to test for
#indice is the index for pred var
#lambda is regularisation param
#alpha is determine Lasso(alpha = 1), Ridge(alpha = 0), EN(alpha = 0.5)
#lambda grid is lambdas to test for

# Function 3: CV with date support
lasso.cv.lambda <- function(Y, nprev, indice = 1, alpha = 1, 
                            lambda_grid = 10^seq(-6, 1, length.out = 50),
                            date_col = NULL) {
  
  # Store results for each lambda
  cv_results <- data.frame(
    lambda = lambda_grid,
    rmse = NA,
    mae = NA
  )
  
  cat("Testing", length(lambda_grid), "lambda values...\n")
  
  # Test each lambda
  for(i in seq_along(lambda_grid)) {
    lambda <- lambda_grid[i]
    
    # Run rolling window with this lambda
    result <- lasso.rolling.window(Y, nprev, indice, lambda = lambda, 
                                   alpha = alpha, date_col = date_col)
    
    # Store errors
    cv_results$rmse[i] <- result$errors["rmse"]
    cv_results$mae[i] <- result$errors["mae"]
    
    if(i %% 10 == 0) {
      cat("Completed", i, "of", length(lambda_grid), "lambdas\n")
    }
  }
  
  # Find best lambda (minimum RMSE)
  best_idx <- which.min(cv_results$rmse)
  best_lambda <- cv_results$lambda[best_idx]
  
  cat("\nBest lambda:", best_lambda, "with RMSE:", cv_results$rmse[best_idx], "\n")
  
  # Refit with best lambda to get final model
  best_model <- lasso.rolling.window(Y, nprev, indice, lambda = best_lambda, 
                                     alpha = alpha, date_col = date_col)
  
  # Plot CV results
  par(mfrow = c(1, 2))
  plot(log10(cv_results$lambda), cv_results$rmse, type = "b", 
       xlab = "log10(Lambda)", ylab = "RMSE", main = "CV: RMSE vs Lambda")
  abline(v = log10(best_lambda), col = "red", lty = 2)
  
  plot(log10(cv_results$lambda), cv_results$mae, type = "b",
       xlab = "log10(Lambda)", ylab = "MAE", main = "CV: MAE vs Lambda")
  abline(v = log10(best_lambda), col = "red", lty = 2)
  par(mfrow = c(1, 1))
  
  return(list(
    "best_lambda" = best_lambda,
    "best_model" = best_model,
    "cv_results" = cv_results
  ))
}