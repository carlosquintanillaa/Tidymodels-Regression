# Data Preparation ----
library(tidyverse)
library(tidymodels)

# Read data set, create partitions, define metrics.

yield <- read_csv("yield.csv")

set.seed(1234)

yield_split <- initial_split(yield, prop = 0.80, strata="Yield")

yield_train <- training(yield_split)
yield_test <- testing(yield_split)

folds <- vfold_cv(yield_train, v = 10, strata="Yield")
metric <- metric_set(rmse,rsq,mae)

# Model Ridge, Mixture = 0, Get lambda_max_ridge ----

Ridge_spec <-
  linear_reg(
    engine = "glmnet",
    penalty = 1,
    mixture = 0
  ) 

Ridge_rec <-
  recipe(Yield ~ ., data = yield_train) %>%
  step_normalize(all_numeric_predictors()) %>% 
  step_dummy(all_nominal_predictors()) 

Ridge_wflow <- 
  workflow() %>% 
  add_model(Ridge_spec) %>%
  add_recipe(Ridge_rec)

# Extract engine
a=fit(Ridge_wflow,data=yield_train)
lambda_max_ridge = a$fit$fit$fit$lambda[1]

# Model Lasso, Mixture = 1, Get lambda_max_lasso ----

Lasso_spec <-
  linear_reg(
    engine = "glmnet",
    penalty = 1,
    mixture = 1
  ) 

Lasso_rec <-
  recipe(Yield ~ ., data = yield_train) %>%
  step_normalize(all_numeric_predictors()) %>% 
  step_dummy(all_nominal_predictors()) 

Lasso_wflow <- 
  workflow() %>% 
  add_model(Lasso_spec) %>%
  add_recipe(Lasso_rec)

# Extract engine
b=fit(Lasso_wflow,data=yield_train)
lambda_max_lasso = b$fit$fit$fit$lambda[1]


# Now estimate the optimal lambda for Ridge Regression ----

Ridge_spec <-
  linear_reg(
    engine = "glmnet",
    penalty = tune("lambda"),
    mixture = 0
  ) 

Ridge_rec <-
  recipe(Yield ~ ., data = yield_train) %>%
  step_normalize(all_numeric_predictors()) %>% 
  step_dummy(all_nominal_predictors()) 

Ridge_wflow <- 
  workflow() %>% 
  add_model(Ridge_spec) %>%
  add_recipe(Ridge_rec)


Ridge_grid = expand_grid(lambda=c(0,10^(seq(log10(0.0001*lambda_max_ridge), log10(lambda_max_ridge), length.out = 30))))

Ridge_res <- 
  tune_grid(
    Ridge_wflow,
    resamples = folds,
    metrics = metric,
    grid = Ridge_grid
  )

collect_metrics(Ridge_res) 

show_best(Ridge_res,metric="rmse")

best_Ridge_params <- select_best(Ridge_res,metric="rmse")
best_Ridge_params

# 'Best' Ridge Regression 

best_Ridge_wflow <- finalize_workflow(Ridge_wflow,best_Ridge_params)

best_Ridge_fit <- last_fit(best_Ridge_wflow, yield_split)

collect_metrics(best_Ridge_fit)

# Predicting new observations

best_Ridge_model <- fit(best_Ridge_wflow,data=yield_train)
predicciones_best_Ridge <- predict(best_Ridge_model,new_data=yield_test)
predicciones_best_Ridge

# Now estimate the optimal lambda for Lasso Regression ----

Lasso_spec <-
  linear_reg(
    engine = "glmnet",
    penalty = tune("lambda"),
    mixture = 1
  ) 

Lasso_rec <-
  recipe(Yield ~ ., data = yield_train) %>%
  step_normalize(all_numeric_predictors()) %>% 
  step_dummy(all_nominal_predictors()) 

Lasso_wflow <- 
  workflow() %>% 
  add_model(Lasso_spec) %>%
  add_recipe(Lasso_rec)


Lasso_grid = expand_grid(lambda=c(0,10^(seq(log10(0.0001*lambda_max_lasso), log10(lambda_max_lasso), length.out = 30))))

Lasso_res <- 
  tune_grid(
    Lasso_wflow,
    resamples = folds,
    metrics = metric,
    grid = Lasso_grid
  )

collect_metrics(Lasso_res) 

show_best(Lasso_res,metric="rmse")

best_Lasso_params <- select_best(Lasso_res,metric="rmse")
best_Lasso_params

# 'Best' Lasso Regression 

best_Lasso_wflow <- finalize_workflow(Lasso_wflow,best_Lasso_params)

best_Lasso_fit <- last_fit(best_Lasso_wflow, yield_split)

collect_metrics(best_Lasso_fit)

# Predicting new observations

best_Lasso_model <- fit(best_Lasso_wflow,data=yield_train)
predicciones_best_Lasso <- predict(best_Lasso_model,new_data=yield_test)
predicciones_best_Lasso

