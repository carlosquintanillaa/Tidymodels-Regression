library(tidyverse)
library(tidymodels)

# Read data set, create partitions, define metrics

diamonds <- read_csv("diamantes.csv") 

# Draw a random sample of 2000 to try the models

set.seed(1234)

 diamonds <- diamonds %>%    
  sample_n(2000)

diamonds_split <- initial_split(diamonds, prop = 0.80, strata="price")

diamonds_train <- training(diamonds_split)
diamonds_test <- testing(diamonds_split)

folds <- vfold_cv(diamonds_train, v = 10, strata="price")

metric <- metric_set(rmse,rsq,mae)

# Model Linear Regression

linreg_spec <-
  linear_reg(
    engine = "lm"
  ) 

linreg_rec <-
  recipe(price ~ ., data = diamonds_train) %>%
  step_dummy(all_nominal_predictors()) 

linreg_wflow <- 
  workflow() %>% 
  add_model(linreg_spec) %>%
  add_recipe(linreg_rec)

linreg_res <- 
  fit_resamples(
    linreg_wflow,
    resamples = folds,
    metrics = metric
  )

collect_metrics(linreg_res)

show_best(linreg_res,metric="rmse")

# 'Best' Linear Regression 

best_linreg_fit <- last_fit(linreg_wflow, diamonds_split)

collect_metrics(best_linreg_fit)

# Predicting new observations

best_linreg_model <- fit(linreg_wflow,data=diamonds_train)
predicciones_linreg <- predict(best_linreg_model,new_data=diamonds_test)


