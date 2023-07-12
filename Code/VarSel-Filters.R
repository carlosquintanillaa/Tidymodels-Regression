library(tidyverse)
library(tidymodels)
library(colino)

lmt0 = read_csv("lmt-ejemplo.csv")

# Graficos

ggplot(lmt0,aes(ingreso,educacion,color=y)) +
  geom_point()

lmt0 %>% 
  filter(sexo=="mujer") %>% 
  ggplot(aes(ingreso,educacion,color=y)) +
    geom_point()

lmt0 %>% 
  filter(sexo=="hombre") %>% 
  ggplot(aes(ingreso,educacion,color=y)) +
  geom_point()

# Recode dependent variable as factor. Reorder levels.

lmt0$y = as_factor(lmt0$y)
lmt0$y = fct_relevel(lmt0$y,c("yes","no"))

# Define function to add noise ----

add_noise <- function(data, numvar, mean = 0, sd = 1) {
  nn <- nrow(data)
  matriz <- matrix(rnorm(nn * numvar, mean, sd), nrow = nn, ncol = numvar)
  colnames(matriz) <- paste0("R", 1:numvar)
  return(bind_cols(data, as_tibble(matriz)))
}

set.seed(1234)

# Agregar ruido a la hoja original (30 variables ruido)

lmt = add_noise(lmt0,30)

# Crear particiones y folds de CV y definir metricas de evaluacio

lmt_split = initial_split(lmt, prop = 0.80, strata="y")

lmt_train = training(lmt_split)
lmt_test = testing(lmt_split)

folds <- vfold_cv(lmt_train, v = 10, strata="y")
metric <- metric_set(roc_auc,accuracy)

# Models : KNN ----

knn_spec <-
  nearest_neighbor(
    mode = "classification", 
    neighbors = 1,
    engine = "kknn"
  ) 

knn_rec <-
  recipe(y ~ ., data = lmt_train) %>%
  step_select_boruta(all_predictors(),outcome="y") %>% 
  step_normalize(all_numeric_predictors()) %>% 
  step_dummy(all_nominal_predictors()) 

#prepped = prep(knn_rec)
#juiced = juice(prepped)

knn_wflow <- 
  workflow() %>% 
  add_model(knn_spec) %>%
  add_recipe(knn_rec)

knn_res <- 
  fit_resamples(
    knn_wflow,
    resamples = folds,
    metrics = metric
  )

collect_metrics(knn_res)
