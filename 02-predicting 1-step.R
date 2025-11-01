### Doing the model 
# Assessment metrics

#RMSE
RMSE <- function(pred, truth) sqrt(mean((truth - pred)^2))

# Directional accuracy
dir_acc <- function(pred, actual) mean(sign(pred) == sign(actual))

# Trading simulation
trade_sim <- function(pred, actual) cumprod(c(1, ifelse(pred > 0, 1+actual, 1-actual)))

trade_sim_sum <- function(pred, actual) cumsum(c(1, ifelse(pred > 0, actual, -actual)))

trade_sim_cost <- function(pred, actual, cost = 0.001) {
  positions <- sign(pred)
  trades <- c(TRUE, diff(positions) != 0)
  returns <- ifelse(pred > 0, 1+actual, 1-actual) - ifelse(trades, cost, 0)
  cumprod(c(1, returns))
}

# FOr reference , almost 60% of obs are positive so we are looking to beat that
ntest = 247
yy = test$ep
yy_sign = mean(sign(yy)>0)
yy_sign = max(c(yy_sign, 1-yy_sign))

# RW Baseline
rw = embed(data$ep, 2)[, 2]
rw = tail(rw, ntest)

#Plotting
plot(yy,type = "l")
lines(rw, col = "red")

all_preds = data.frame(
  actual = yy,
  rw = rw
)

model_results = data.frame(
  model= "RW",
  RMSE=RMSE(rw, yy),
  dir_acc=dir_acc(rw, yy),
  return=trade_sim(rw, yy)[248],
  return_w_cost=trade_sim_cost(rw, yy)[248],
  return_sum = trade_sim_sum(rw, yy)[248]
)

#historical mean
rm = cumsum(data$ep)/1:nrow(data)
rm = tail(rm, ntest)

#Plotting
plot(yy,type = "l")
lines(rm, col = "red")

add_results <- function(df, pred, name){
  df <- rbind(df, data.frame(
    model= name,
    RMSE=RMSE(pred, yy),
    dir_acc=dir_acc(pred, yy),
    return=trade_sim(pred, yy)[248],
    return_w_cost=trade_sim_cost(pred, yy)[248],
    return_sum = trade_sim_sum(pred, yy)[248]    
  ))
}

all_preds = cbind(all_preds, rm)
model_results = add_results(model_results, rm, "Historic Mean")


# Starting of with best subset selection, we select the model with the lowest AIC
tic()
bssel = regsubsets(ep~.-yyyymm, data = train, nvmax = 19, method = "exhaustive")
toc()

sumbss = summary(bssel)
sumbss


# by BIC we see that the model w 5 regressors has the smallest BIC error, corpr, infl, svar, skvw and avgcor
# the model that is the best based on BIC
plot(bssel, scale = "bic")

# We use full model to estimate var as 495/21 is ~23 so not too bad
ntrain = 495
varest=sumbss$rss[19]/(ntrain-20) #estimate error variance of the model with k=21


#construct the IC with this estimate (note seq() used to generate a sequence from 1 to 11 regressors to plug into the formula):
BICL = sumbss$rss/ntrain + log(ntrain)*varest*((seq(1,19,1))/ntrain)
AICL = sumbss$rss/ntrain + 2*varest*((seq(1,19,1))/ntrain)

kbicl=which.min(BICL) #BIC choice

kaicl=which.min(AICL)  #AIC choice (AIC proportional to Cp and so ranking is the same)


#Compute IC for the models using sigma0 estimate:
BIC0 = sumbss$rss/ntrain + log(ntrain)*(sumbss$rss[kbicl]/(ntrain-kbicl-1))*((seq(1,19,1))/ntrain)
AIC0 = sumbss$rss/ntrain + 2*(sumbss$rss[kaicl]/(ntrain-kaicl-1))*((seq(1,19,1))/ntrain)

#Select best models
k0bic = which.min(BIC0)
k0aic = which.min(AIC0)

#Obtain IC estimates with sigma 1 estimate:
BIC1 = sumbss$rss/ntrain + log(ntrain)*(sumbss$rss[k0bic]/(ntrain-k0bic-1))*((seq(1,19,1))/ntrain)
AIC1 = sumbss$rss/ntrain + 2*(sumbss$rss[k0aic]/(ntrain-k0aic-1))*((seq(1,19,1))/ntrain)

k1bic = which.min(BIC1)
k1aic = which.min(AIC1)

# As we expect, due to the increase in penalisation for BIC where we multiply complexity by log(k), BIC favours the simpler model corpr, infl, svar, skvw and avgcor
# While AIC favours a more complex model of 10 predictors, corpr, b/m, infl, svar, ogap, wtexas, skvw dtoy, dtoat, d/e

test.mat = model.matrix(ep~.-yyyymm, data = data)

temp.coef = names(coef(bssel, id = k1bic))
lm_BIC_data = test.mat[, temp.coef]

temp.coef = names(coef(bssel, id = k1aic))
lm_AIC_data = test.mat[, temp.coef]

lm_AIC_data = cbind(data$ep, lm_AIC_data)
lm_BIC_data = cbind(data$ep, lm_BIC_data)

source("custom-func-lm.R")

lm_AIC = lm.rolling.window(lm_AIC_data, ntest, 1)
lm_BIC = lm.rolling.window(lm_BIC_data, ntest, 1)

lm_AIC$coef
lm_BIC$coef


all_preds = cbind(all_preds, lm_AIC$pred)
all_preds = cbind(all_preds, lm_BIC$pred)
model_results = add_results(model_results, lm_AIC$pred, "Subset Selection - AIC")
model_results = add_results(model_results, lm_BIC$pred, "Subset Selection - BIC")

source("custom-func-lasso.R")

alpha = 1

test.mat = scale(model.matrix(ep~.-yyyymm, data = data)[, 2:20])
lasso_data = cbind(ep=data$ep, test.mat)
lasso_cv_data = lasso_data[1:(nrow(lasso_data)-ntest), ]

#of the 495 take the last half so 247 for CV
lasso_CV = lasso.cv.lambda(lasso_cv_data, ntest, indice = 1, alpha = alpha, 
                           lambda_grid = 10^seq(-6, 1, length.out = 50), date_col = data$yyyymm[(nrow(data)-ntest*2):(nrow(data)-ntest)])
#We use RMSE as main metric in determining for CV and from the graph can see that best lambda is 0.01930698

#prediction
lasso_test = lasso.rolling.window(lasso_data, ntest, indice = 1, lasso_CV$best_lambda, alpha = alpha, date_col = data$yyyymm[(nrow(data)-ntest):nrow(data)])

all_preds = cbind(all_preds, lasso_test$pred)
model_results = add_results(model_results, lasso_test$pred, "Lasso")

#for ridge
alpha = 0

#of the 495 take the last half so 247 for CV
ridge_CV = lasso.cv.lambda(lasso_cv_data, ntest, indice = 1, alpha = alpha, #just change this can alrdy
                           lambda_grid = 10^seq(-6, 1, length.out = 50), date_col = data$yyyymm[(nrow(data)-ntest*2):(nrow(data)-ntest)])
#We use RMSE as main metric in determining for CV and from the graph can see that best lambda is 0.01930698

#prediction
ridge_test = lasso.rolling.window(lasso_data, ntest, indice = 1, ridge_CV$best_lambda, alpha = alpha, date_col = data$yyyymm[(nrow(data)-ntest):nrow(data)])

all_preds = cbind(all_preds, ridge_test$pred)
model_results = add_results(model_results, ridge_test$pred, "Ridge")


# For EN
alpha = 0.5

#of the 495 take the last half so 247 for CV
EN_CV = lasso.cv.lambda(lasso_cv_data, ntest, indice = 1, alpha = alpha, #just change this can alrdy
                           lambda_grid = 10^seq(-6, 1, length.out = 50), date_col = data$yyyymm[(nrow(data)-ntest*2):(nrow(data)-ntest)])
#We use RMSE as main metric in determining for CV and from the graph can see that best lambda is 0.01930698

#prediction
EN_test = lasso.rolling.window(lasso_data, ntest, indice = 1, ridge_CV$best_lambda, alpha = alpha, date_col = data$yyyymm[(nrow(data)-ntest):nrow(data)])

all_preds = cbind(all_preds, EN_test$pred)
model_results = add_results(model_results, EN_test$pred, "EN")

# due to the biasedness of LASSO, we try post-Lasso

source("custom-func-plasso.R")

alpha = 1

#of the 495 take the last half so 247 for CV
plasso_CV = plasso.cv.lambda(lasso_cv_data, ntest, indice = 1, alpha = alpha, 
                           lambda_grid = 10^seq(-5, 0, length.out = 50), date_col = data$yyyymm[(nrow(data)-ntest*2):(nrow(data)-ntest)])
#We use RMSE as main metric in determining for CV and from the graph can see that best lambda is 0.01930698

#prediction
plasso_test = plasso.rolling.window(lasso_cv_data, ntest, indice = 1, lasso_CV$best_lambda, alpha = alpha, date_col = data$yyyymm[(nrow(data)-ntest):nrow(data)])

all_preds = cbind(all_preds, plasso_test$pred)
model_results = add_results(model_results, plasso_test$pred, "Post Lasso")

# PCR
source("custom-func-pcr.R")

#of the 495 take the last half so 247 for CV
pcr_cv <- pcr.cv.ncomp(lasso_cv_data, nprev = ntest, indice = 1, date_col = data$yyyymm[(nrow(data)-ntest*2):(nrow(data)-ntest)])

#We use RMSE as main metric in determining for CV and from the graph can see that best lambda is 0.01930698

#prediction
pcr_test <- pcr.rolling.window(lasso_cv_data, nprev = ntest, indice = 1, ncomp = pcr_cv$best_ncomp, date_col = data$yyyymm[(nrow(data)-ntest):nrow(data)])

final_loadings = pcr_test$loadings[ntest, 1:19, 1]
barplot(abs(final_loadings/sum(final_loadings)))

all_preds = cbind(all_preds, pcr_test$pred)
model_results = add_results(model_results, pcr_test$pred, "PCR")

# PLS
source("custom-func-pls.R")

pls_cv <- pls.cv.ncomp(lasso_cv_data, nprev = ntest, indice = 1, date_col = data$yyyymm[(nrow(data)-ntest*2):(nrow(data)-ntest)])

#prediction
pls_test <- pls.rolling.window(lasso_cv_data, nprev = ntest, indice = 1, ncomp = pls_cv$best_ncomp, date_col = data$yyyymm[(nrow(data)-ntest):nrow(data)])

all_preds = cbind(all_preds, pls_test$pred)
model_results = add_results(model_results, pls_test$pred, "PLS")

# RF takes a long time
rf_data = model.matrix(ep~.-yyyymm, data = data)[, 2:20] # Dont need to scale
source("custom-func-rf.R")

rf_cv <- rf.cv.ntree(rf_data, nprev = ntest, indice = 1, date_col = data$yyyymm[(nrow(data)-ntest*2):(nrow(data)-ntest)])

#prediction
rf_test = rf.rolling.window(rf_data, nprev = ntest, indice = 1, ntree = rf_cv$best_ntree, date_col = data$yyyymm[(nrow(data)-ntest):nrow(data)])

all_preds = cbind(all_preds, rf_test$pred)
model_results = add_results(model_results, rf_test$pred, "RF")


# XGB
source("custom-func-xgb.R")

xgb_cv <- xgb.cv.nrounds(rf_data, nprev = ntest, indice = 1, date_col = data$yyyymm[(nrow(data)-ntest*2):(nrow(data)-ntest)])

#prediction
xgb_test = xgb.rolling.window(rf_data, nprev = ntest, indice = 1, nrounds = xgb_cv$best_nrounds, date_col = data$yyyymm[(nrow(data)-ntest):nrow(data)])

all_preds = cbind(all_preds, xgb_test$pred)
model_results = add_results(model_results, xgb_test$pred, "XGB")

# ensemble
#simple average of all past models
ensemble_pred = rowSums(all_preds[, 4:ncol(all_preds)])/(ncol(all_preds)-3)
all_preds = cbind(all_preds, ensemble_pred)
model_results = add_results(model_results, ensemble_pred, "ensemble")
