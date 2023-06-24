# Load libraries, read data and create partitions  ----

library(tidyverse)
library(tidymodels)
library(stacks)
library(discrim)

# Read data set.

bank <- read_csv("bank.csv") 

# Recode dependent variable as factor. Reorder levels.

bank$y = as_factor(bank$y)
bank$y = fct_relevel(bank$y,c("yes","no"))

# Take random sample to speed the process of running the program

set.seed(1234)
bank <- bank %>%
  sample_n(2000)

## Create partitions, define metrics.

bank_split = initial_split(bank, prop = 0.80, strata="y")

bank_train = training(bank_split)
bank_test = testing(bank_split)

folds <- vfold_cv(bank_train, v = 10, strata="y")
metric <- metric_set(roc_auc,accuracy)

ctrl_grid <- control_stack_grid()
ctrl_res <- control_stack_resamples()

# Models : Logistic Regression ----

logistic_reg_spec <-
  logistic_reg(
    engine="glm"
) 

logistic_reg_rec <-
  recipe(y ~ ., data = bank_train) %>%
  step_dummy(all_nominal_predictors()) 

logistic_reg_wflow <- 
  workflow() %>%
  add_model(logistic_reg_spec) %>%
  add_recipe(logistic_reg_rec)

logistic_reg_res <- 
  fit_resamples(
    logistic_reg_wflow,
    resamples = folds,
    metrics = metric,
    control = ctrl_res
  )

collect_metrics(logistic_reg_res)

# Models : Naive Bayes ----

nb_spec <- naive_Bayes(
  mode = "classification",
  engine = "naivebayes",
  Laplace = 0.5
)

nb_rec  <-
  recipe(y ~ ., data = bank_train)

nb_wflow <- 
  workflow() %>%
  add_model(nb_spec) %>%
  add_recipe(nb_rec)

nb_res <- 
  fit_resamples(
    nb_wflow,
    resamples = folds,
    metrics = metric,
    control = ctrl_res
  )

collect_metrics(nb_res)

# Models : KNN ----

knn_spec <-
  nearest_neighbor(
    mode = "classification", 
    neighbors = tune("k"),
    engine = "kknn"
  ) 

knn_rec <-
  recipe(y ~ ., data = bank_train) %>%
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
    mode = "classification",
    tree_depth = tune("depth"),
    engine = "rpart"
  ) 

tree_rec <-
  recipe(y ~ ., data = bank_train) 

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
    mode="classification",
    trees = tune("trees"),
    mtry = tune("mtry"),
    engine = "ranger"
  ) 

rf_rec <-
  recipe(y ~ ., data = bank_train) 

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
  mode = "classification",
  trees = tune("trees"),
  tree_depth = tune("depth"),
  learn_rate = tune("eta"),
  engine = "xgboost"
) 

xgb_rec <-
  recipe(y ~ ., data = bank_train) %>%
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

bank_st <- 
  stacks() %>%
  add_candidates(logistic_reg_res) %>%
  add_candidates(nb_res) %>% 
  add_candidates(knn_res) %>%
  add_candidates(tree_res) %>% 
  add_candidates(rf_res) %>% 
  add_candidates(xgb_res)

# AUCs all members of stack

aucs_members <- bank_st %>% 
  as_tibble() %>% 
  select(contains(".pred_yes")) %>% 
  map(roc_auc_vec, truth = bank_train$y) %>%
  as_tibble() 

View(t(aucs_members))

# Fit the stack ----

bank_st <- 
  bank_st %>% 
  blend_predictions(metric = metric_set(roc_auc)) %>% 
  fit_members()

bank_st
autoplot(bank_st,type="weights")

collect_parameters(bank_st,"xgb_res")

# Predict Test Set : AUC  & Accuracy ----

bank_test_preds <- 
  bank_test %>%
  bind_cols(predict(bank_st,type="prob", .),predict(bank_st, type="class", .)) %>% 
  select(y,starts_with(".pred"))

bank_test_preds  

roc_auc(
  bank_test_preds,
  truth = y,
  estimate = .pred_yes
)

accuracy(
  bank_test_preds,
  truth = y,
  estimate = .pred_class
)
