### Read in the Data

df <- read_excel("Data2024.xlsx", sheet = 2, col_types = rep("numeric", ncol(read_excel("Data2024.xlsx", sheet = 2, n_max = 1))))


##EDA
# based on preliminary analysis and looking at the paper and the data set we chose these data from the papers
data = df %>% select(1, 6:16, 18, 22, 24, 25, 34, 37, 38, 42:44, 46:49, 55) %>% mutate(ep = retx-Rfree) %>% select(c(-2, -9))

#yyyymm is the index and ep is our dependent value
colnames(data)

adf_results <- imap(
  data %>% select(-yyyymm),  # Exclude index and target
  ~{
    col_data <- na.omit(.x)
    y_data <- data$ep
    
    # ADF test
    adf_result <- tryCatch(adf.test(col_data), error = function(e) NULL)
    adf_p <- if (!is.null(adf_result)) adf_result$p.value else NA
    
    ts_col <- ts(col_data, start = c(1958, 2), end = c(2019, 12), frequency = 12)
    plot.ts(ts_col, main = .y, guess_max = 1000)
    
    tibble(
      column_name = .y,
      adf_p_value = adf_p,
    )
  }
) %>% bind_rows()

# Looking at the Results, AAA, BAA, lty, tbl all are non stationary, thus we considered tms which is lty-tbl and dfr which is corpr-ltr
adf.test(na.omit(df$tms))
adf.test(na.omit(df$dfr))

# plotting dp and dy considering the construction we guess high correlation and decide to remove dy to reduce multicollinearity
dp <- ts(na.omit(df$'d/p'))
plot.ts(dp, main = "d/p")

dy <- ts(na.omit(df$'d/y'))
plot.ts(dy, main = "d/y")

cor(dp[2:1848], dy, use = "complete.obs", method = "pearson")

#final data
data = df %>% select(1, 6, 10, 11, 13, 14, 16, 18: 20, 22, 24, 25, 34, 37, 38, 42:44, 46:49, 55) %>% mutate(ep = retx-Rfree) %>% select(c(-'retx', -'Rfree')) %>% na.omit() %>% dplyr::filter(yyyymm<202000) 

summary(data)

# We currently have data from 1958-02 to 2019-12 which is where we cut off as we hope to avoid the covid period
summary(data$yyyymm)

# Checking for persistent things variable like inflation
# Does not look too non-stationary, but a possible change in mean 
infl <- ts(na.omit(df$infl), start = c(1913, 3), frequency = 12)
plot.ts(infl, main = "infl", xlab = "Time")

# Looking at the y data we see that it looks stationary
ggplot(data, aes(x=yyyymm, ep)) +
  geom_line()

# lead y_t to ensure temporal consisteny, aka predict y_t w x_t-1
data = data %>% arrange(yyyymm) %>% mutate(ep=lead(ep, 2), yyyymm = lead(yyyymm, 2)) %>% na.omit()


### Splitting into training and test data
train = data %>% dplyr::slice(c(1:494))
test = data %>% dplyr::slice(c(495:742))

### EDA so plot stuff like linear correlations
lag_corr <- imap(
  train %>% select(-yyyymm, -ep),  # Exclude index and target
  ~{
    col_data <- .x
    y_data <- train$ep
    
    # Correlation with y_t and x_t-1
    corr <- abs(cor(col_data, y_data, use = "complete.obs"))
    
    # Correlations with x_{t-2} to x_{t-6}
    lagged_corrs <- map_dbl(c(1, 2, 5, 11), function(k) {
      lagged_x <- dplyr::lag(col_data, k)
      abs(cor(lagged_x, y_data, use = "complete.obs"))
    })
    
    tibble(
      column_name = .y,
      correlation_with_y = corr,
      abs_cor_lag2 = lagged_corrs[1],
      abs_cor_lag3 = lagged_corrs[2],
      abs_cor_lag6 = lagged_corrs[3],
      abs_cor_lag12 = lagged_corrs[4]
    )
  }
) %>% bind_rows()

lm = lm(ep~.-yyyymm, data = train)
vif(lm)
#Given the high VIF for d/p and e/p we consider including d/e instead

train = train %>% mutate(`d/e` = `d/p`-`e/p`) %>% select(-`d/p`, -`e/p`)
test = test %>% mutate(`d/e` = `d/p`-`e/p`) %>% select(-`d/p`, -`e/p`)
data = data %>% mutate(`d/e` = `d/p`-`e/p`) %>% select(-`d/p`, -`e/p`)

lm = lm(ep~.-yyyymm-`d/e`, data = train)
vif(lm)

lm = lm(ep~.-yyyymm-ygap, data = train)
vif(lm)

# remove ygap as correlation btw y and past lags of ygap is max 2% and max VIF is lower at around 6
# Benefits of removing include stability of weights, making causal interpretation more reliable, lower variance of models and possibly better OOS performance and 

train = train %>% select(-"ygap")
test = test %>% select(-"ygap")
data = data %>% select(-"ygap")