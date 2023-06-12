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


# Model : Random Forest 

rf_spec <- 
  rand_forest(
    mode="regression",
    trees = tune("trees"),
    mtry = tune("mtry"),
    engine = "ranger"
  ) 

rf_rec <-
  recipe(price ~ ., data = diamonds_train) 

rf_wflow <- 
  workflow() %>% 
  add_model(rf_spec) %>%
  add_recipe(rf_rec)

rf_grid = expand.grid(trees=c(10,100,500),mtry=c(3,4,5))

rf_res <- 
  tune_grid(
    rf_wflow,
    resamples = folds,
    metrics = metric,
    grid = rf_grid
  )

collect_metrics(rf_res)
autoplot(rf_res)

show_best(rf_res,metric="rmse")

# Best Random Forest

best_rf_spec <- 
  rand_forest(
    mode = "regression",
    trees = 100,
    mtry = 5,
    engine = "ranger"
  ) 

best_rf_wflow <- 
  workflow() %>% 
  add_model(best_rf_spec) %>%
  add_recipe(rf_rec)


best_rf_fit <- last_fit(best_rf_wflow, diamonds_split)

collect_metrics(best_rf_fit)

# Predicting new observations

best_rf_model <- fit(best_rf_wflow,data=diamonds_train)
predicciones_rf <- predict(best_rf_model,new_data=diamonds_test)



