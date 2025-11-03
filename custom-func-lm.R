# Adapted from func-ar.r from Medeiros et al. (2019)

#One for forming forecasts using lm model, which will be called
#on each iteration of the rolling window forecasting exercise.

#The other one for producing the series of h-step forecasts using rolling window.

#Inputs for the function:

#1) Data matrix Y: includes all variables

#2) indice - index for dependent variable: 1 for CPI inflation, 2 for PCE inflation

runlm <- function(Y, indice) {
  
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
  
  # Fit OLS model
  model <- lm(y_train ~ ., data = as.data.frame(X_train))
  
  # Predict on the last row
  pred <- predict(model, newdata = as.data.frame(X_test))
  
  return(list(model = model, pred = as.numeric(pred)))
}


#This function will repeatedly call the previous function in the rolling window h-step forecasting

#Inputs for the function:

#1) Data matrix Y: includes all variables

#2) nprev - number of out-of-sample observations (at the end of the sample)

#3) indice - index for dependent variable: 1 for CPI inflation, 2 for PCE inflation


lm.rolling.window <- function(Y, nprev, indice = 1) {
  
  save.coef <- matrix(NA, nprev, ncol(Y))
  save.pred <- matrix(NA, nprev, 1)
  
  for(i in nprev:1) {
    Y.window <- Y[(1 + nprev - i):(1+nrow(Y) - i), ]
    lm.model <- runlm(Y.window, indice)
    save.coef[(1 + nprev - i), ] <- coef(lm.model$model)
    coef(lm.model$model)
    save.pred[(1 + nprev - i), ] <- lm.model$pred
  }
  
  real <- tail(Y[, indice], nprev)
  plot(real, type = "l")
  lines(c(rep(NA, length(real) - nprev), save.pred), col = "red")
  
  rmse <- sqrt(mean((tail(real, nprev) - save.pred)^2))
  mae <- mean(abs(tail(real, nprev) - save.pred))
  errors <- c("rmse" = rmse, "mae" = mae)
  
  return(list("pred" = save.pred, "coef" = save.coef, "errors" = errors))
}

