#################################################################
# Predicting Extreme Weather in Madrid
# Clean analysis script (PCA + PCR + Linear Regression)
# Assumes working directory is the project root:
#   - data/a2_data_group_22.csv
#################################################################

# --- 0. Packages -----------------------------------------------------------

if (!require("pacman")) install.packages("pacman")
pacman::p_load(
  dplyr, tibble, readr, tidyr,
  ggplot2, boot, pls
)

# --- 1. Data loading & preparation ----------------------------------------

# Read data
df <- read.csv("data/a2_data_group_22.csv")

# Clean column names
names(df) <- make.names(names(df))

# Rename variables for clarity
df <- df %>%
  rename(
    mean_temp           = Mean.temperature...C.,
    max_temp            = Max..temperature...C.,
    min_temp            = Min..temperature...C.,
    perceived_mean_temp = Perceived.mean.temperature...C.,
    perceived_max_temp  = Perceived.max..temperature...C.,
    perceived_min_temp  = Perceived.min..temperature...C.,
    max_wind_speed      = Max..wind.speed..km.h.,
    max_wind_gusts      = Max..wind.gusts..km.h.,
    radiation           = Shortwave.radiation.sum..MJ.m2.,
    wind_direction      = Dominant.wind.direction....,
    evapotranspiration  = Reference.evapotranspiration..mm.,
    daylight_duration   = Daylight.duration..s.,
    sunshine_duration   = Sunshine.duration..s.,
    precipitation       = Precipitation.sum..mm.,
    snowfall            = Snowfall.sum..mm.,
    precipitation_hours = Precipitation.hours..h.,
    rain                = Rain.sum..mm.
  )

# --- 2. Supervised learning structure (today -> tomorrow) -----------------

# Predict tomorrow's max temperature using today's weather
X <- df[1:(nrow(df) - 1), ]           # predictors: all but last row
y <- df$max_temp[2:nrow(df)]         # response: max temp of next day

# Keep only numeric predictors (excluding potential ID variables)
numeric_X <- X[sapply(X, is.numeric) & names(X) != "Location.ID"]

# --- 3. Correlation analysis ----------------------------------------------

correlations <- sapply(numeric_X, function(col) cor(col, y, use = "complete.obs"))

# Optional: correlation table
cor_table <- tibble(
  Variable     = names(correlations),
  Correlation  = as.numeric(correlations)
) %>%
  mutate(
    Interpretation = case_when(
      Correlation >  0.9 ~ "Very strong positive",
      Correlation >  0.7 ~ "Strong positive",
      Correlation >  0.5 ~ "Moderate positive",
      Correlation < -0.9 ~ "Very strong negative",
      Correlation < -0.7 ~ "Strong negative",
      Correlation < -0.5 ~ "Moderate negative",
      TRUE               ~ "Negligible"
    )
  )

# Example: inspect highest correlations
cor_table %>% arrange(desc(abs(Correlation))) %>% head()

# --- 4. Train–test split (chronological) ----------------------------------

n          <- length(y)
train_size <- floor(0.8 * n)

X_train <- numeric_X[1:train_size, ]
y_train <- y[1:train_size]

X_test  <- numeric_X[(train_size + 1):n, ]
y_test  <- y[(train_size + 1):n]

# --- 5. PCA on training data ----------------------------------------------

res <- princomp(X_train, cor = TRUE, scores = TRUE)

# Eigenvalues and explained variance
ev             <- res$sdev^2
var_explained  <- ev / sum(ev)
cum_var        <- cumsum(var_explained)

pca_var <- data.frame(
  PC                 = seq_along(ev),
  Variance_Explained = var_explained,
  Cumulative_Variance = cum_var
)

# Cumulative variance plot
plot(
  pca_var$PC, pca_var$Cumulative_Variance,
  type = "b", pch = 19,
  xlab = "Principal Component",
  ylab = "Cumulative Proportion of Variance Explained",
  main = "Cumulative Variance Explained by PCs"
)
abline(h = 0.8, col = "red", lty = 2)

# Scree plot with Kaiser threshold
plot(
  ev,
  type = "b", pch = 19,
  xlab = "Principal Component",
  ylab = "Eigenvalue",
  main = "Scree Plot"
)
abline(h = 1, col = "red", lty = 2)

# Biplot (PC1 vs PC2)
rownames(res$scores) <- rep(".", res$n.obs)
biplot(
  res, pc.biplot = TRUE,
  scale = 1, las = 1,
  col = c(rgb(0, 0, 0.5, 0.25), rgb(0.5, 0, 0))
)

# --- 6. Bootstrap CI for proportion of variance explained by PC1 ----------

prop_pc1_hat <- ev[1] / sum(ev)

X_use <- as.data.frame(X_train)
# remove columns with missing or zero variance
X_use <- X_use[, colSums(!is.na(X_use)) == nrow(X_use), drop = FALSE]
const_cols <- sapply(X_use, function(z) sd(z) == 0)
if (any(const_cols)) X_use <- X_use[, !const_cols, drop = FALSE]
X_mat <- as.matrix(X_use)

boot_pc1_prop <- function(data, idx) {
  x <- data[idx, , drop = FALSE]
  ev_b <- try(princomp(x, cor = TRUE)$sdev^2, silent = TRUE)
  if (inherits(ev_b, "try-error")) return(NA_real_)
  ev_b[1] / sum(ev_b)
}

set.seed(123)
B        <- 2000
fit.boot <- boot(data = X_mat, statistic = boot_pc1_prop, R = B)

prop_vec <- fit.boot$t[, 1]
prop_vec <- prop_vec[is.finite(prop_vec)]

ci_pc1 <- quantile(prop_vec, probs = c(0.025, 0.975))

hist(
  prop_vec,
  breaks = 30,
  main = "Bootstrap CI — PC1 Variance Ratio",
  xlab  = "Proportion of Variance (PC1)",
  col   = "gray"
)
abline(v = ci_pc1,     col = "darkgreen", lwd = 2)
abline(v = prop_pc1_hat, col = "red",      lwd = 2)

ci_pc1

# --- 7. Variable importance from PC1 --------------------------------------

load_pc1 <- res$loadings[, 1]
fit_pc1  <- as.numeric(load_pc1)^2

best_table <- data.frame(
  Variable    = names(load_pc1),
  Loading_PC1 = as.numeric(load_pc1),
  Fit_by_PC1  = fit_pc1
) %>%
  arrange(desc(Fit_by_PC1))

head(best_table, 10)

# --- 8. Build modelling datasets ------------------------------------------

train_df <- data.frame(y = as.numeric(y_train), X_train)
test_df  <- data.frame(y = as.numeric(y_test),  X_test)

# --- 9. PCR with cross-validation -----------------------------------------

set.seed(42)
pcr_cv <- pcr(
  y ~ .,
  data       = train_df,
  scale      = TRUE,
  validation = "CV",
  segments   = 5,
  segment.type = "consecutive"
)

# RMSEP across number of components
rm_obj   <- RMSEP(pcr_cv, estimate = "CV")
cv_rmse  <- drop(rm_obj$val[1, 1, -1])   # exclude 0-component model
k_min    <- which.min(cv_rmse)

# Plot CV RMSE vs components
plot(
  cv_rmse,
  type = "b", pch = 19,
  xlab = "Number of Components",
  ylab = "CV RMSE",
  main = "CV RMSE for PCR"
)
abline(v = k_min, col = "red", lwd = 2)

# Predictions on test set using optimal number of components
pcr_pred_test <- as.numeric(predict(pcr_cv, newdata = test_df, ncomp = k_min))

# --- 10. Linear regression benchmark --------------------------------------

lm_model    <- lm(y ~ ., data = train_df)
lm_pred_test <- predict(lm_model, newdata = test_df)

# --- 11. Metrics functions -------------------------------------------------

rmse <- function(actual, pred) sqrt(mean((actual - pred)^2))
mae  <- function(actual, pred) mean(abs(actual - pred))
r2   <- function(actual, pred) 1 - sum((actual - pred)^2) /
  sum((actual - mean(actual))^2)

# Metrics for PCR (optimal k_min)
pcr_rmse <- rmse(y_test, pcr_pred_test)
pcr_mae  <- mae(y_test,  pcr_pred_test)
pcr_r2   <- r2(y_test,   pcr_pred_test)

# Metrics for linear regression
lm_rmse <- rmse(y_test, lm_pred_test)
lm_mae  <- mae(y_test,  lm_pred_test)
lm_r2   <- r2(y_test,   lm_pred_test)

model_comparison <- data.frame(
  Model = c("PCR (k_min)", "Linear Regression"),
  RMSE  = c(pcr_rmse,      lm_rmse),
  MAE   = c(pcr_mae,       lm_mae),
  R2    = c(pcr_r2,        lm_r2)
)

model_comparison

# --- 12. PCR sensitivity analysis: 3, 4, 5 components ---------------------

pcr_3 <- pcr(y ~ ., data = train_df, scale = TRUE, ncomp = 3)
pcr_4 <- pcr(y ~ ., data = train_df, scale = TRUE, ncomp = 4)
pcr_5 <- pcr(y ~ ., data = train_df, scale = TRUE, ncomp = 5)

pcr_pred_3 <- as.numeric(predict(pcr_3, newdata = test_df, ncomp = 3))
pcr_pred_4 <- as.numeric(predict(pcr_4, newdata = test_df, ncomp = 4))
pcr_pred_5 <- as.numeric(predict(pcr_5, newdata = test_df, ncomp = 5))

rmse_3 <- rmse(y_test, pcr_pred_3)
mae_3  <- mae(y_test,  pcr_pred_3)
r2_3   <- r2(y_test,   pcr_pred_3)

rmse_4 <- rmse(y_test, pcr_pred_4)
mae_4  <- mae(y_test,  pcr_pred_4)
r2_4   <- r2(y_test,   pcr_pred_4)

rmse_5 <- rmse(y_test, pcr_pred_5)
mae_5  <- mae(y_test,  pcr_pred_5)
r2_5   <- r2(y_test,   pcr_pred_5)

pcr_sensitivity <- data.frame(
  Model = c("PCR (3 comps)", "PCR (4 comps)", "PCR (5 comps)"),
  RMSE  = c(rmse_3,          rmse_4,          rmse_5),
  MAE   = c(mae_3,           mae_4,           mae_5),
  R2    = c(r2_3,            r2_4,            r2_5)
)

pcr_sensitivity

# Add linear regression to comparison
all_models <- rbind(
  pcr_sensitivity,
  data.frame(
    Model = "Linear Regression",
    RMSE  = lm_rmse,
    MAE   = lm_mae,
    R2    = lm_r2
  )
)

all_models

# --- 13. Decile-level MSE comparison --------------------------------------

set.seed(123)

test_df_mse <- data.frame(
  max_temp  = y_test,
  pred_pcr3 = pcr_pred_3,
  pred_pcr4 = pcr_pred_4,
  pred_pcr5 = pcr_pred_5,
  pred_lm   = lm_pred_test
) %>%
  mutate(
    max_temp_jittered = max_temp + rnorm(n(), mean = 0, sd = 0.01)
  )

# Compute deciles based on jittered max temperature
decile_breaks <- quantile(
  test_df_mse$max_temp_jittered,
  probs = seq(0, 1, 0.1),
  na.rm = TRUE
)

test_df_mse <- test_df_mse %>%
  mutate(
    decile = cut(
      max_temp_jittered,
      breaks = decile_breaks,
      labels = 1:10,
      include.lowest = TRUE
    )
  )

# MSE per decile per model
mse_df <- test_df_mse %>%
  group_by(decile) %>%
  summarise(
    mse_pcr3 = mean((max_temp - pred_pcr3)^2, na.rm = TRUE),
    mse_pcr4 = mean((max_temp - pred_pcr4)^2, na.rm = TRUE),
    mse_pcr5 = mean((max_temp - pred_pcr5)^2, na.rm = TRUE),
    mse_lm   = mean((max_temp - pred_lm)^2,   na.rm = TRUE),
    .groups  = "drop"
  ) %>%
  pivot_longer(
    cols      = starts_with("mse_"),
    names_to  = "model",
    values_to = "mse"
  ) %>%
  mutate(
    model = dplyr::recode(
      model,
      "mse_pcr3" = "PCR (3 comps)",
      "mse_pcr4" = "PCR (4 comps)",
      "mse_pcr5" = "PCR (5 comps)",
      "mse_lm"   = "Linear Regression"
    ),
    decile = as.numeric(as.character(decile))
  )

# Plot MSE across deciles
ggplot(mse_df, aes(x = decile, y = mse, color = model)) +
  geom_line(linewidth = 1) +
  geom_point(size = 2) +
  labs(
    title = "MSE Across Deciles: PCR (3, 4, 5 comps) vs Linear Regression",
    x     = "Decile (1 = Coldest Days, 10 = Hottest Days)",
    y     = "Mean Squared Error (MSE)",
    color = "Model"
  ) +
  scale_x_continuous(breaks = 1:10) +
  theme_minimal()

#################################################################
# End of script
#################################################################
