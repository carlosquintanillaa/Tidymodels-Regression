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

# Model Regression Trees 

tree_spec <-
  decision_tree(
    mode = "regression", 
    tree_depth = tune("depth"),
    engine = "rpart"
  ) 

tree_rec <-
  recipe(price ~ ., data = diamonds_train) 

tree_wflow <- 
  workflow() %>% 
  add_model(tree_spec) %>%
  add_recipe(tree_rec)

tree_grid <- expand.grid(depth=c(3,5,10,20))

tree_res <- 
  tune_grid(
    tree_wflow,
    resamples = folds,
    metrics = metric,
    grid = tree_grid
  )

collect_metrics(tree_res)
autoplot(tree_res)

show_best(tree_res,metric="rmse")

# Best Tree

best_tree_spec <-
  decision_tree(
    mode = "regression", 
    tree_depth = 5,
    engine = "rpart"
  ) 

best_tree_wflow <- 
  workflow() %>% 
  add_model(best_tree_spec) %>%
  add_recipe(tree_rec)

best_tree_fit <- last_fit(best_tree_wflow, diamonds_split)

collect_metrics(best_tree_fit)

# Predicting new observations

best_tree_model <- fit(best_tree_wflow,data=diamonds_train)
predicciones_tree <- predict(best_tree_model,new_data=diamonds_test)

# Graph best tree 

# Extract information from engine 'rpart'

a=extract_fit_engine(best_tree_model)

library(rpart.plot)

rpart.plot(a,box.palette="RdBu",shadow.col="grey",nn=TRUE,roundint = FALSE)

