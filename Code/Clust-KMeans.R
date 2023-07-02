library(tidymodels)
library(tidyclust)
library(GGally)
library(janitor)

# Create function to denormalize data (if needed/wanted)

denormalize <- function(cst,means,sds){
  denormalized <- cbind(cst[1], t(t(cst[-1])*unlist(sds) + unlist(means))) %>% as_tibble()
  denormalized
}

# Convert to tibble

iris_tb <- as_tibble(iris)
iris_tb

# Compute vector of means & sds

means <- iris_tb %>% 
  select(-Species) %>% 
  summarise(across(everything(), mean))

sds <- iris_tb %>% 
  select(-Species) %>% 
  summarise(across(everything(), sd)) 

# Create model

set.seed(1234)

kmeans_rec <- recipe(~ .,data=iris_tb ) %>%
  step_normalize(all_numeric_predictors()) %>% 
  step_rm(Species) 

kmeans_spec <- k_means(
  num_clusters = 2,
  engine = "stats"
)

kmeans_wflow <- 
  workflow() %>% 
  add_model(kmeans_spec) %>% 
  add_recipe(kmeans_rec)

# Fit model 

kmeans_model <- fit(kmeans_wflow,data=iris_tb)
kmeans_model

# Standardized Centroids

centroids_standardized = extract_centroids(kmeans_model)
centroids_standardized

centroids_standardized %>% 
  ggparcoord(columns = 2:5, groupColumn = 1, scale = "globalminmax") 

# Centroids in original units

centroids_original <- denormalize(centroids_standardized,means,sds)
centroids_original

centroids_original %>% 
  ggparcoord(columns = 2:5, groupColumn = 1, scale = "globalminmax") 

# Create assignment variable and add it to the original dataset

assignment <- extract_cluster_assignment(kmeans_model) 

new_iris_tb <- iris_tb %>% 
  bind_cols(assignment)

new_iris_tb

new_iris_tb %>% 
  tabyl(.cluster)

new_iris_tb %>% 
  tabyl(.cluster,Species)
