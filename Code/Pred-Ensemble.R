# Load libraries, read data and create partitions  ----
library(tidyverse)
library(tidymodels)
library(stacks)

diamonds <- read_csv("diamantes.csv") 

set.seed(1234)

diamonds <- diamonds %>%    
  sample_n(2000)

diamonds_split <- initial_split(diamonds, prop = 0.80, strata="price")

diamonds_train <- training(diamonds_split)
diamonds_test <- testing(diamonds_split)

folds <- vfold_cv(diamonds_train, v = 10, strata="price")
metric <- metric_set(rmse,rsq,mae)

ctrl_grid <- control_stack_grid()
ctrl_res <- control_stack_resamples()

# Models : Linear Regression ---- 

linreg_spec <-
  linear_reg(engine="lm") 

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
    metrics = metric,
    control = ctrl_res
  )

collect_metrics(linreg_res)


# Models : KNN ----

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

knn_grid = expand.grid(k=c(1,5,10,30))

knn_res <- 
  tune_grid(
    knn_wflow,
    resamples = folds,
    metrics = metric,
    grid = knn_grid,
    control = ctrl_grid
  )

collect_metrics(knn_res)

# Models : Decision Trees ----

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

tree_grid = expand.grid(depth=c(3,5,10,20))

tree_res <- 
  tune_grid(
    tree_wflow,
    resamples = folds,
    metrics = metric,
    grid = tree_grid,
    control = ctrl_grid
  )

collect_metrics(tree_res)

# Models : Random Forest ----

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
    grid = rf_grid,
    control = ctrl_grid
  )

collect_metrics(rf_res)

# Models : Xgboost ----

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
    grid = xgb_grid,
    control = ctrl_grid
  )

collect_metrics(xgb_res)

# Create the stack ----

diamonds_st <- 
  stacks() %>%
  add_candidates(knn_res) %>%
  add_candidates(linreg_res) %>%
  add_candidates(tree_res) %>% 
  add_candidates(rf_res) %>% 
  add_candidates(xgb_res)

# RMSEs all members of stack on the training set

diamonds_st %>% 
  as_tibble() %>% 
  map(rmse_vec, truth = diamonds_train$price) %>%
  as_tibble() 


# Fit the stack & show the solution ----

diamonds_st <- 
  diamonds_st %>% 
  blend_predictions(metric = metric_set(rmse)) %>% 
  fit_members()

diamonds_st

autoplot(diamonds_st,type="weights")

collect_parameters(diamonds_st,"xgb_res")

# Performance on Test Set

member_preds <- 
  diamonds_test %>%
  select(price) %>%
  bind_cols(predict(diamonds_st, diamonds_test, members = TRUE))

member_preds

# Graph Predicted vs Actual Values----

ggplot(member_preds, aes(x = .pred, y = price)) +
  geom_point() + 
  coord_obs_pred() +
  geom_abline(intercept = 0,slope=1)

# Correlations

cor(member_preds$price,member_preds$.pred)
cor(member_preds)

# Ensemble Performance

ensemble_performance = member_preds %>% 
  map(rmse_vec, truth = member_preds$price) %>% 
  as_tibble()

ensemble_performance

# End ----












