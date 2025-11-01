#Adapted from Medeiros

#Y is the data
#indice is the index for pred var
#lambda is regularisation param
#alpha is determine Lasso(alpha = 1), Ridge(alpha = 0), EN(alpha = 0.5)

# Function 1: Train on first n-1 rows, predict last row with specified lambda
runplasso <- function(Y, indice, lambda, alpha = 1) {
  
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
  
  # Step 1: Fit LASSO model with specified lambda
  lasso_model <- glmnet(X_train, y_train, alpha = alpha, lambda = lambda)
  
  # Step 2: Extract coefficients and find non-zero ones (excluding intercept)
  lasso_coefs <- as.numeric(coef(lasso_model, s = lambda))
  intercept <- lasso_coefs[1]
  beta <- lasso_coefs[-1]
  
  # Find which variables were selected (non-zero coefficients)
  selected <- which(beta != 0)
  
  # Step 3: Fit OLS on selected variables
  if (length(selected) == 0) {
    # If no variables selected, use intercept-only model
    warning("No variables selected by LASSO. Using intercept-only model.")
    ols_model <- lm(y_train ~ 1)
    pred <- predict(ols_model, newdata = data.frame(intercept = 1))
    
    # Create coefficient vector (intercept + zeros)
    final_coefs <- c(mean(y_train), rep(0, ncol(X_train)))
    
  } else {
    # Fit OLS on selected variables
    X_train_selected <- X_train[, selected, drop = FALSE]
    X_test_selected <- X_test[, selected, drop = FALSE]
    
    # Create data frame for lm
    train_data <- data.frame(y = y_train, X_train_selected)
    test_data <- data.frame(X_test_selected)
    colnames(test_data) <- colnames(train_data)[-1]
    
    # Fit OLS
    ols_model <- lm(y ~ ., data = train_data)
    
    # Generate prediction
    pred <- predict(ols_model, newdata = test_data)
    
    # Create full coefficient vector (intercept + all variables)
    final_coefs <- rep(0, ncol(X_train) + 1)
    final_coefs[1] <- coef(ols_model)[1]  # Intercept
    final_coefs[selected + 1] <- coef(ols_model)[-1]  # Selected variable coefficients
  }
  
  return(list(
    "model" = ols_model, 
    "pred" = as.numeric(pred),
    "selected_vars" = selected,
    "lasso_model" = lasso_model,
    "coefficients" = final_coefs
  ))
}

#Y is the data
#nprev is number of obs to test for
#indice is the index for pred var
#lambda is regularisation param
#alpha is determine Lasso(alpha = 1), Ridge(alpha = 0), EN(alpha = 0.5)
#datecol to plot dates

# Function 2: Rolling window forecasting with specified lambda (with date support)
plasso.rolling.window <- function(Y, nprev, indice = 1, lambda, alpha = 1, 
                                  date_col = NULL, n_date_labels = 8) {
  
  save.coef <- matrix(NA, nprev, ncol(Y))
  save.pred <- matrix(NA, nprev, 1)
  save.selected <- list()  # Store which variables were selected each period
  
  for(i in nprev:1) {
    Y.window <- Y[(1 + nprev - i):(1 + nrow(Y) - i), ]
    plasso.model <- runplasso(Y.window, indice, lambda = lambda, alpha = alpha)
    save.coef[(1 + nprev - i), ] <- plasso.model$coefficients
    save.pred[(1 + nprev - i), ] <- plasso.model$pred
    save.selected[[1 + nprev - i]] <- plasso.model$selected_vars
  }
  
  # Get actual values
  real <- Y[, indice]
  real_test <- tail(real, nprev)
  
  # Handle dates if provided
  if (!is.null(date_col)) {
    dates <- tail(date_col, nprev)
    
    # Convert YYYYMM to Date format
    if (is.numeric(dates)) {
      year <- floor(dates / 100)
      month <- dates %% 100
      dates <- as.Date(paste(year, month, "01", sep = "-"))
    } else if (!inherits(dates, "Date")) {
      dates <- as.Date(as.character(dates), format = "%Y%m")
    }
    
    # Plot with dates on x-axis
    plot(dates, real_test, type = "l", 
         main = paste("Post-LASSO (OLS), Lambda =", round(lambda, 6)),
         ylab = "Value", xlab = "Date", xaxt = "n", lwd = 1.5)
    lines(dates, save.pred, col = "red", lwd = 1.5)
    
    # Smart axis labeling
    n_ticks <- min(n_date_labels, length(dates))
    tick_positions <- dates[round(seq(1, length(dates), length.out = n_ticks))]
    axis.Date(1, at = tick_positions, format = "%Y-%m")
    
  } else {
    # Plot without dates
    plot(real_test, type = "l", 
         main = paste("Post-LASSO (OLS), Lambda =", round(lambda, 6)),
         ylab = "Value", xlab = "Time Period", lwd = 1.5)
    lines(save.pred, col = "red", lwd = 1.5)
  }
  
  legend("topright", legend = c("Actual", "Predicted"), 
         col = c("black", "red"), lty = 1, lwd = 1.5)
  
  # Compute errors
  rmse <- sqrt(mean((real_test - save.pred)^2))
  mae <- mean(abs(real_test - save.pred))
  errors <- c("rmse" = rmse, "mae" = mae)
  
  # Calculate average number of variables selected
  n_selected <- sapply(save.selected, length)
  avg_selected <- mean(n_selected)
  
  return(list(
    "pred" = save.pred, 
    "coef" = save.coef, 
    "errors" = errors,
    "selected_vars" = save.selected,
    "avg_n_selected" = avg_selected
  ))
}

#Y is the data
#nprev is number of obs to test for
#indice is the index for pred var
#lambda is regularisation param
#alpha is determine Lasso(alpha = 1), Ridge(alpha = 0), EN(alpha = 0.5)
#lambda grid is lambdas to test for

# Function 3: CV with date support
plasso.cv.lambda <- function(Y, nprev, indice = 1, alpha = 1, 
                             lambda_grid = 10^seq(-6, 1, length.out = 50),
                             date_col = NULL) {
  
  # Store results for each lambda
  cv_results <- data.frame(
    lambda = lambda_grid,
    rmse = NA,
    mae = NA,
    avg_n_selected = NA
  )
  
  cat("Testing", length(lambda_grid), "lambda values...\n")
  
  # Test each lambda
  for(i in seq_along(lambda_grid)) {
    lambda <- lambda_grid[i]
    
    # Run rolling window with this lambda
    result <- plasso.rolling.window(Y, nprev, indice, lambda = lambda, 
                                    alpha = alpha, date_col = date_col)
    
    # Store errors and selection info
    cv_results$rmse[i] <- result$errors["rmse"]
    cv_results$mae[i] <- result$errors["mae"]
    cv_results$avg_n_selected[i] <- result$avg_n_selected
    
    if(i %% 10 == 0) {
      cat("Completed", i, "of", length(lambda_grid), "lambdas\n")
    }
  }
  
  # Find best lambda (minimum RMSE)
  best_idx <- which.min(cv_results$rmse)
  best_lambda <- cv_results$lambda[best_idx]
  
  cat("\nBest lambda:", best_lambda, 
      "with RMSE:", cv_results$rmse[best_idx],
      "| Avg variables selected:", round(cv_results$avg_n_selected[best_idx], 1), "\n")
  
  # Refit with best lambda to get final model
  best_model <- plasso.rolling.window(Y, nprev, indice, lambda = best_lambda, 
                                      alpha = alpha, date_col = date_col)
  
  # Plot CV results
  par(mfrow = c(1, 3))
  plot(log10(cv_results$lambda), cv_results$rmse, type = "b", 
       xlab = "log10(Lambda)", ylab = "RMSE", main = "CV: RMSE vs Lambda")
  abline(v = log10(best_lambda), col = "red", lty = 2)
  
  plot(log10(cv_results$lambda), cv_results$mae, type = "b",
       xlab = "log10(Lambda)", ylab = "MAE", main = "CV: MAE vs Lambda")
  abline(v = log10(best_lambda), col = "red", lty = 2)
  
  plot(log10(cv_results$lambda), cv_results$avg_n_selected, type = "b",
       xlab = "log10(Lambda)", ylab = "Avg # Variables", 
       main = "CV: Model Size vs Lambda")
  abline(v = log10(best_lambda), col = "red", lty = 2)
  par(mfrow = c(1, 1))
  
  return(list(
    "best_lambda" = best_lambda,
    "best_model" = best_model,
    "cv_results" = cv_results
  ))
}