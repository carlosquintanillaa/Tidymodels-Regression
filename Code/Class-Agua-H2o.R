
library(tidyverse)
library(tidymodels)
library(h2o)
library(agua)

# Initialize h2o cluster

h2o.init()

# Read dataset & recode dependent variable as a factor (instead of a character)

bank <- read_csv("bank.csv") 

bank$y = as_factor(bank$y)
bank$y = fct_relevel(bank$y,c("yes","no"))

# Create the data split
set.seed(1234)

bank_split = initial_split(bank, prop = 0.80, strata="y")

bank_train = training(bank_split)
bank_test = testing(bank_split)

# Model: Auto-model (h2o)

h2o_spec <-
  auto_ml() %>%
  set_engine("h2o", max_runtime_secs = 300) %>%
  set_mode("classification")

h2o_rec <-
  recipe(y ~ ., data = bank_train)  

h2o_wflow <-
  workflow() %>%
  add_model(h2o_spec) %>%
  add_recipe(h2o_rec)

h2o_fit <- fit(h2o_wflow, data = bank_train)

collect_metrics(h2o_fit)

rank_results(h2o_fit) %>%
  filter(.metric == "auc") %>%
  arrange(rank)

# Predict test set observations and evaluate performance

pred_class_h2o <- predict(h2o_fit, new_data= bank_test, type="class")
pred_prob_h2o <- predict(h2o_fit, new_data = bank_test, type="prob")

roc_auc_vec(truth=bank_test$y,estimate=pred_prob_h2o$.pred_yes)
accuracy_vec(truth=bank_test$y,estimate=pred_class_h2o$.pred_class)

# Shut down the cluster

h2o.shutdown(prompt = FALSE)
