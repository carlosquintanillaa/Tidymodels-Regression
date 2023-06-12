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

# Model KNN 

knn_spec <-
  nearest_neighbor(
    mode = "regression", 
    neighbors = tune("k"),
    engine = "kknn"
  ) 

knn_rec <-
  recipe(price ~ ., data = diamonds_train) %>%
  step_normalize(all_numeric_predictors()) %>% 
  step_dummy(all_nominal_predictors()) 

knn_wflow <- 
  workflow() %>% 
  add_model(knn_spec) %>%
  add_recipe(knn_rec)

knn_grid <- expand.grid(k=c(1,5,10,30))

knn_res <- 
  tune_grid(
    knn_wflow,
    resamples = folds,
    metrics = metric,
    grid = knn_grid
  )

collect_metrics(knn_res)
autoplot(knn_res)

show_best(knn_res,metric="rmse")

# Best KNN 

best_knn_spec <-
  nearest_neighbor(
    mode = "regression", 
    neighbors = 10,
    engine = "kknn"
  ) 

best_knn_wflow <- 
  workflow() %>% 
  add_model(best_knn_spec) %>%
  add_recipe(knn_rec)

best_knn_fit <- last_fit(best_knn_wflow, diamonds_split)

collect_metrics(best_knn_fit)

# Predicting new observations

best_knn_model <- fit(best_knn_wflow,data=diamonds_train)
predicciones_knn <- predict(best_knn_model,new_data=diamonds_test)


