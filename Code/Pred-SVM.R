library(tidyverse)
library(tidymodels)

# Read data set, create partitions, define metrics.

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

# Model KNN 

svm_spec <-
  svm_rbf(
    mode = "regression", 
    cost = tune("c"),
    rbf_sigma = tune("gamma"),
    engine = "kernlab"
  ) 

svm_rec <-
  recipe(price ~ ., data = diamonds_train) %>%
  step_normalize(all_numeric_predictors()) %>% 
  step_dummy(all_nominal_predictors()) 

svm_wflow <- 
  workflow() %>% 
  add_model(svm_spec) %>%
  add_recipe(svm_rec)

svm_grid = expand.grid(c=c(0.125,0.5,1,2,8),gamma=2^c(-5,-3,-1,1,3))

svm_res <- 
  tune_grid(
    svm_wflow,
    resamples = folds,
    metrics = metric,
    grid = svm_grid
  )

collect_metrics(svm_res)
autoplot(svm_res)

show_best(svm_res,metric="rmse")

# Best KNN 

best_svm_spec <-
  svm_rbf(
    mode = "regression", 
    cost = 8,
    rbf_sigma = 0.0312,
    engine = "kernlab"
  ) 

best_svm_wflow <- 
  workflow() %>% 
  add_model(best_svm_spec) %>%
  add_recipe(svm_rec)

best_svm_fit <- last_fit(best_svm_wflow, diamonds_split)

collect_metrics(best_svm_fit)

# Predicting new observations

best_svm_model <- fit(best_svm_wflow,data=diamonds_train)
predicciones_svm <- predict(best_svm_model,new_data=diamonds_test)



