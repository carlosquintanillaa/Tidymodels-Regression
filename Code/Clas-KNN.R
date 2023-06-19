library(tidyverse)
library(tidymodels)

# Read data set, create partitions, define metrics.

bank <- read_csv("bank.csv")

bank$y <- as_factor(bank$y)
bank$y <- fct_relevel(bank$y,c("yes","no"))

set.seed(1234)
bank <- bank %>%
  sample_n(2000)

bank_split <- initial_split(bank, prop=0.8, strata="y")

bank_train <- training(bank_split)
bank_test <- testing(bank_split)

fold <- vfold_cv(bank_train, v=10, strata="y")
metric <- metric_set(roc_auc,accuracy)

# Model : KNN

knn_spec <- 
  nearest_neighbor(
    mode = "classification",
    neighbors = tune("k"),
    engine = "kknn"
  )

knn_rec <- 
  recipe(y ~ ., data=bank_train) %>% 
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
    resamples = fold,
    metrics = metric,
    grid = knn_grid
  )

collect_metrics(knn_res)
show_best(knn_res,"roc_auc")

# Select best model automatically. Estimate performance on test set

best_knn_params <- select_best(knn_res,"roc_auc")
best_knn_wflow <- finalize_workflow(knn_wflow,best_knn_params)

best_knn_fit <- last_fit(best_knn_wflow,bank_split )
collect_metrics(best_knn_fit)

# Predicting new observations (classes & probabilities)

best_knn_model <- fit(best_knn_wflow,data=bank_train )

pred_class_knn <- predict(best_knn_model, new_data= bank_test, type="class")
pred_prob_knn <- predict(best_knn_model, new_data = bank_test, type="prob")
