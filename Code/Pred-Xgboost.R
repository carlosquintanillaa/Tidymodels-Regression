library(tidyverse)
library(tidymodels)

# Read dataset, create partitions, define metrics.

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

# Models : xgboost

xgb_spec <- boost_tree(
  mode = "regression",
  trees = tune("trees"),
  tree_depth = tune("depth"),
  learn_rate = tune("eta"),
  engine = "xgboost"
) 

xgb_rec <-
  recipe(price ~ ., data = diamonds_train) %>%
  step_dummy(all_nominal_predictors())

xgb_wflow <- 
  workflow() %>% 
  add_model(xgb_spec) %>%
  add_recipe(xgb_rec)

xgb_grid <- expand.grid(depth=c(2,4,6,8),trees=c(10,100,500),eta=c(0.01,0.1,0.2))

xgb_res <- 
  tune_grid(
    xgb_wflow,
    resamples = folds,
    metrics = metric,
    grid = xgb_grid
  )

collect_metrics(xgb_res)
autoplot(xgb_res)

show_best(xgb_res,metric="rmse")

# Best Xgboost

best_xgb_spec <- boost_tree(
  mode = "regression",
  trees = 500,
  tree_depth = 2,
  learn_rate = 0.2,
  engine = "xgboost"
  )

best_xgb_wflow <- 
  workflow() %>% 
  add_model(best_xgb_spec) %>%
  add_recipe(xgb_rec)

best_xgb_fit <- last_fit(best_xgb_wflow, diamonds_split)

collect_metrics(best_xgb_fit)

# Predicting new observations

best_xgb_model <- fit(best_xgb_wflow,data=diamonds_train)
predicciones_xgb <- predict(best_xgb_model,new_data=diamonds_test)
