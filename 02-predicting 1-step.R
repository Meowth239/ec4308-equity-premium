### Doing the model 
# Assessment metrics

set.seed(4308)

#RMSE
RMSE <- function(pred, truth) sqrt(mean((truth - pred)^2))

# Directional accuracy
dir_acc <- function(pred, actual) mean(sign(pred) == sign(actual))

# Trading simulation
trade_sim <- function(pred, actual) cumprod(c(1, ifelse(pred > 0, 1+actual, 1-actual)))

trade_sim_sum <- function(pred, actual) cumsum(ifelse(pred > 0, actual, -actual))

trade_sim_cost <- function(pred, actual, cost = 0.001) {
  positions <- sign(pred)
  trades <- c(TRUE, diff(positions) != 0)
  returns <- ifelse(pred > 0, 1+actual, 1-actual) - ifelse(trades, cost, 0)
  cumprod(c(1, returns))
}

PT_test <- function(predicted, actual) {
  n <- length(actual)
  threshold = mean(actual)
  actual_up <- as.numeric(actual >= threshold)
  pred_up <- as.numeric(predicted >= threshold)
  
  # Calculate observed proportions
  P_x <- mean(actual_up)      # Proportion of actual "up"
  P_y <- mean(pred_up)        # Proportion of predicted "up"
  
  # Calculate observed hit rate
  correct_direction <- as.numeric(actual_up == pred_up)
  P_hat <- mean(correct_direction)
  
  # Calculate expected hit rate under independence
  P_star <- P_x * P_y + (1 - P_x) * (1 - P_y)
  
  # Create the series Z_t = I_t - P_star
  # where I_t = 1 if correct direction, 0 otherwise
  Z_t <- correct_direction - P_star
    
  # Fit regression of Z_t on constant (testing if mean differs from 0)
  model <- lm(Z_t ~ 1)
  
  se_hac <- sqrt(NeweyWest(model, lag = floor(n^(1/3))))
  
  # Calculate test statistic
  # We're testing H0: E[Z_t] = 0, i.e., mean(Z_t) = 0
  test_stat <- mean(Z_t) / se_hac
  
  return (test_stat)
}

# For reference , almost 60% of obs are positive so we are looking to beat that
yy = data$ep[(nrow(data)-ntest+1):nrow(data)]
yy_sign = mean(sign(yy)>0)
yy_sign = max(c(yy_sign, 1-yy_sign))


squared_loss <- function(pred) {
  (yy-pred)^2
}

#historical mean
hm = cumsum(data$ep)/1:nrow(data)
hm = tail(rm, ntest)

#Plotting
plot(yy,type = "l", main = "Actual Vs Historical Mean", xaxt = 'n')
axis(side = 1, at=c(8, 68, 128, 188, 247), labels = c(2000, 2005, 2010, 2015, 2020))
lines(hm, col = "red")
mean(yy)

all_preds = data.frame(
  actual = yy,
  hm = hm
)

model_results = data.frame(
  model= "Historic Mean",
  RMSE=RMSE(hm, yy),
  dir_acc=dir_acc(hm, yy),
  return=trade_sim(hm, yy)[ntest+1],
  return_w_cost=trade_sim_cost(hm, yy)[ntest+1],
  return_sum = trade_sim_sum(hm, yy)[ntest],
  PT_test = PT_test(hm, yy)
)

add_results <- function(df, pred, name){
  df <- rbind(df, data.frame(
    model= name,
    RMSE=RMSE(pred, yy),
    dir_acc=dir_acc(pred, yy),
    return=trade_sim(pred, yy)[ntest+1],
    return_w_cost=trade_sim_cost(pred, yy)[ntest+1],
    return_sum = trade_sim_sum(pred, yy)[ntest],
    PT_test = PT_test(pred, yy)
  ))
}

#squared loss for dm test, we mainly use hm as the benchmark
squared_loss_df = data.frame(hm = squared_loss(hm))


# Starting of with best subset selection, we select the model with the lowest AIC
tic()
bssel = regsubsets(ep~.-yyyymm, data = data[1:(nrow(data)-ntest), ], nvmax = 19, method = "exhaustive")
toc()

sumbss = summary(bssel)
sumbss


# by BIC we see that the model w 5 regressors has the smallest BIC error, corpr, infl, svar, skvw and avgcor
# the model that is the best based on BIC
plot(bssel, scale = "bic")

# We use full model to estimate var as 495/21 is ~23 so not too bad
ntrain = nrow(data)-247
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
print("Variables chosen by BIC")
temp.coef
lm_BIC_data = test.mat[, temp.coef]

temp.coef = names(coef(bssel, id = k1aic))
print("Variables chosen by AIC")
temp.coef
lm_AIC_data = test.mat[, temp.coef]

lm_AIC_data = cbind(data$ep, lm_AIC_data)
lm_BIC_data = cbind(data$ep, lm_BIC_data)

# As expected BIC is much simpler but it doesnt do much better

source("custom-func-lm.R")

lm_AIC = lm.rolling.window(lm_AIC_data, ntest, 1)
lm_BIC = lm.rolling.window(lm_BIC_data, ntest, 1)

#coefficients
lm_AIC$coef
lm_BIC$coef

all_preds = cbind(all_preds, lm_AIC$pred)
all_preds = cbind(all_preds, lm_BIC$pred)
squared_loss_df = cbind(squared_loss_df, squared_loss(lm_AIC$pred))
squared_loss_df = cbind(squared_loss_df, squared_loss(lm_BIC$pred))
model_results = add_results(model_results, lm_AIC$pred, "Subset Selection - AIC")
model_results = add_results(model_results, lm_BIC$pred, "Subset Selection - BIC")

#Plotting
plot(yy,type = "l", main = "Actual Vs Subset Selection  by AIC", xaxt = 'n')
axis(side = 1, at=c(8, 68, 128, 188, 247), labels = c(2000, 2005, 2010, 2015, 2020))
lines(sign(lm_AIC$pred)*yy, col = "red")
lines(sign(hm)*yy, col = "green")

source("custom-func-lasso.R")

#for lasso
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

# lasso coefs for each iteration
lasso_test$coef

lasso_test$observation <- 1:nrow(lasso_test$coef)

# Reshape data to long format for ggplot2
df_long <- melt(lasso_test$coef, id.vars = "observation", 
                variable.name = "variable", 
                value.name = "coefficient")

# Calculate symmetric limits around zero
max_abs <- max(abs(df_long$coefficient), na.rm = TRUE)
limits <- c(-max_abs, max_abs)

# Create heatmap
ggplot(df_long, aes(x = Var1, y = Var2, fill = coefficient)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", mid = "grey", high = "red", 
                       midpoint = 0, limits = limits,
                       name = "Coefficient") +
  theme_minimal() +
  labs(x = "Observation", y = "Variable", 
       title = "Lasso Coefficients Across Rolling Windows") +
  theme(axis.text.x = element_blank(),
        axis.ticks.x = element_blank())

all_preds = cbind(all_prVar1all_preds = cbind(all_preds, lasso_test$pred))
squared_loss_df = cbind(squared_loss_df, squared_loss(lasso_test$pred))
model_results = add_results(model_results, lasso_test$pred, "Lasso")

#for ridge
alpha = 0

#of the 495 take the last half so 247 for CV
ridge_CV = lasso.cv.lambda(lasso_cv_data, ntest, indice = 1, alpha = alpha, #just change this can alrdy
                           lambda_grid = 10^seq(-6, 1, length.out = 50), date_col = data$yyyymm[(nrow(data)-ntest*2):(nrow(data)-ntest)])
#We use RMSE as main metric in determining for CV and from the graph can see that best lambda is 0.01930698

#prediction
ridge_test = lasso.rolling.window(lasso_data, ntest, indice = 1, ridge_CV$best_lambda, alpha = alpha, date_col = data$yyyymm[(nrow(data)-ntest):nrow(data)])

#ridge coef
ridge_test$coef

all_preds = cbind(all_preds, ridge_test$pred)
squared_loss_df = cbind(squared_loss_df, squared_loss(ridge_test$pred))
model_results = add_results(model_results, ridge_test$pred, "Ridge")


# For EN
alpha = 0.5

#of the 495 take the last half so 247 for CV
EN_CV = lasso.cv.lambda(lasso_cv_data, ntest, indice = 1, alpha = alpha, #just change this can alrdy
                           lambda_grid = 10^seq(-6, 1, length.out = 50), date_col = data$yyyymm[(nrow(data)-ntest*2):(nrow(data)-ntest)])
#We use RMSE as main metric in determining for CV and from the graph can see that best lambda is 0.01930698

#prediction
EN_test = lasso.rolling.window(lasso_data, ntest, indice = 1, ridge_CV$best_lambda, alpha = alpha, date_col = data$yyyymm[(nrow(data)-ntest):nrow(data)])

#EN coef
EN_test$coef

all_preds = cbind(all_preds, EN_test$pred)
squared_loss_df = cbind(squared_loss_df, squared_loss(EN_test$pred))
model_results = add_results(model_results, EN_test$pred, "EN")

# due to the biasedness of LASSO, we try post-Lasso

source("custom-func-plasso.R")

alpha = 1

#of the 495 take the last half so 247 for CV
plasso_CV = plasso.cv.lambda(lasso_cv_data, ntest, indice = 1, alpha = alpha, 
                           lambda_grid = 10^seq(-5, 0, length.out = 50), date_col = data$yyyymm[(nrow(data)-ntest*2):(nrow(data)-ntest)])
#We use RMSE as main metric in determining for CV and from the graph can see that best lambda is 0.01930698

#prediction
plasso_test = plasso.rolling.window(lasso_data, ntest, indice = 1, lasso_CV$best_lambda, alpha = alpha, date_col = data$yyyymm[(nrow(data)-ntest):nrow(data)])

# 
plasso_test$selected_vars

all_preds = cbind(all_preds, plasso_test$pred)
squared_loss_df = cbind(squared_loss_df, squared_loss(plasso_test$pred))
model_results = add_results(model_results, plasso_test$pred, "Post Lasso")

# PCR
source("custom-func-pcr.R")

#of the 495 take the last half so 247 for CV
pcr_cv <- pcr.cv.ncomp(lasso_cv_data, nprev = ntest, indice = 1, date_col = data$yyyymm[(nrow(data)-ntest*2):(nrow(data)-ntest)])

#We use RMSE as main metric in determining for CV and from the graph can see that best lambda is 0.01930698

#prediction
pcr_test <- pcr.rolling.window(lasso_data, nprev = ntest, indice = 1, ncomp = pcr_cv$best_ncomp, date_col = data$yyyymm[(nrow(data)-ntest):nrow(data)])

pcr_test$ncomp_used

final_loadings = pcr_test$loadings[ntest, 1:19, 1]
barplot(abs(final_loadings/sum(final_loadings)))

loadings_247 <- pcr_test$loadings[ntest, , ]

# Convert to dataframe
# Rows = variables, Columns = PCs
loadings_df <- as.data.frame(loadings_247)

# Add variable names
loadings_df$variable <- rownames(loadings_df)
# If no rownames, use:
# loadings_df$variable <- paste0("Var", 1:19)

# Reshape to long format
loadings_long <- melt(loadings_df, id.vars = "variable",
                      variable.name = "PC",
                      value.name = "loading")

# Create faceted bar chart
ggplot(loadings_long, aes(x = variable, y = loading)) +
  geom_col(fill = "steelblue") +
  facet_wrap(~ PC, scales = "free_y") +
  theme_minimal() +
  labs(x = "Variable", y = "Loading",
       title = cat("PC Loadings at Observation ", ntest)) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

all_preds = cbind(all_preds, pcr_test$pred)
squared_loss_df = cbind(squared_loss_df, squared_loss(pcr_test$pred))
model_results = add_results(model_results, pcr_test$pred, "PCR")

# PLS
source("custom-func-pls.R")

pls_cv <- pls.cv.ncomp(lasso_cv_data, nprev = ntest, indice = 1, date_col = data$yyyymm[(nrow(data)-ntest*2):(nrow(data)-ntest)])

#prediction
pls_test <- pls.rolling.window(lasso_data, nprev = ntest, indice = 1, ncomp = pls_cv$best_ncomp, date_col = data$yyyymm[(nrow(data)-ntest):nrow(data)])

pls_test$ncomp_used

all_preds = cbind(all_preds, pls_test$pred)
squared_loss_df = cbind(squared_loss_df, squared_loss(pls_test$pred))
model_results = add_results(model_results, pls_test$pred, "PLS")

# RF takes a long time
rf_data = model.matrix(ep~.-yyyymm, data = data)[, 2:20] # Dont need to scale
rf_cv_data = rf_data[1:(nrow(rf_data)-ntest), ]
source("custom-func-rf.R")

rf_cv <- rf.cv.ntree(lasso_cv_data, nprev = ntest, indice = 1, date_col = data$yyyymm[(nrow(data)-ntest*2):(nrow(data)-ntest)])

#prediction
rf_test = rf.rolling.window(lasso_data, nprev = ntest, indice = 1, ntree = rf_cv$best_ntree, date_col = data$yyyymm[(nrow(data)-ntest):nrow(data)])

all_preds = cbind(all_preds, rf_test$pred)
squared_loss_df = cbind(squared_loss_df, squared_loss(rf_test$pred))
model_results = add_results(model_results, rf_test$pred, "RF")


# XGB
source("custom-func-xgb.R")

xgb_cv <- xgb.cv.nrounds(lasso_cv_data, nprev = ntest, indice = 1, date_col = data$yyyymm[(nrow(data)-ntest*2):(nrow(data)-ntest)])

#prediction
xgb_test = xgb.rolling.window(lasso_data, nprev = ntest, indice = 1, nrounds = xgb_cv$best_nrounds, date_col = data$yyyymm[(nrow(data)-ntest):nrow(data)])

all_preds = cbind(all_preds, xgb_test$pred)
squared_loss_df = cbind(squared_loss_df, squared_loss(xgb_test$pred))
model_results = add_results(model_results, xgb_test$pred, "XGB")



# hybrid learning, completely experimental
source("custom-func-hybrid.R")
hybrid_cv_results <- hybrid.cv.params(rf_cv_data, nprev = ntest, indice = 1, date_col = data$yyyymm[(nrow(data)-ntest*2):(nrow(data)-ntest)])

hybrid_test = hybrid.rolling.window(rf_data, nprev = ntest, indice = 1, lambda = hybrid_cv_results$best_lambda, 
                                    nrounds = hybrid_cv_results$best_nrounds,, date_col = data$yyyymm[(nrow(data)-ntest):nrow(data)])

all_preds = cbind(all_preds, hybrid_test$pred)
squared_loss_df = cbind(squared_loss_df, squared_loss(hybrid_test$pred))
model_results = add_results(model_results, hybrid_test$pred, "Hybrid")

# ensemble
#simple average of all past models
ensemble_pred = rowSums(all_preds[, 4:ncol(all_preds)])/(ncol(all_preds)-3)
all_preds = cbind(all_preds, ensemble_pred)
squared_loss_df = cbind(squared_loss_df, squared_loss(ensemble_pred))
model_results = add_results(model_results, ensemble_pred, "ensemble")


for (i in 1:14) {
  temp.ts = ts(yy - all_preds[, i], start = c(1999, 6), end = c(2019, 12), freq = 12)
  plot.ts(temp.ts, ylab = "Actual - Predicted", main = colnames(all_preds)[i])  
}

loss_diff = squared_loss_df[, 1] - squared_loss_df[, -1]
dm_test = c()
for (i in 1:12) {
  temp.ts = ts(loss_diff[, i], start = c(1999, 6), end = c(2019, 12), freq = 12)
  plot.ts(temp.ts, ylab = "Loss Differential", main = colnames(loss_diff)[i])
  dm = lm(loss_diff[, i]~1)
  acf(dm$residuals)
  dm_test = c(dm_test, dm$coefficients/sqrt(NeweyWest(dm,lag=6))) #using general rule of thumb of P^1/3 which is roughly 6
}
model_results = cbind(model_results, dm_test = c(NA, NA, dm_test))


