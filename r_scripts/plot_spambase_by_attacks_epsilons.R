library(tidyverse)
data <- read_csv("saved_results/spambase_mse_by_attack_and_epsilon.csv")

####################### Spambase Original

data2 <- data %>%
  mutate(
    epochs = str_replace(model, "spambase_", ""),
    epochs = as.numeric(epochs),
    epsilon = as.character(epsilon),
    ratio = pmax(distance_class_0, distance_class_1) / pmin(distance_class_0, distance_class_1)
  )

data2 %>%
  ggplot() +
  geom_line(aes(x = epochs, y = ratio, colour = epsilon)) +
  facet_wrap(~attack, scales = "free_y")

data2 %>%
  ggplot() +
  geom_line(aes(x = epochs, y = distance_class_1, colour = epsilon)) +
  facet_wrap(~attack)



####################### Spambase Generic

data <- read_csv("saved_results/spambase_generic_mse_by_attack_and_epsilon.csv")

data2 <- data %>%
  mutate(
    model = str_replace(model, "generic_", ""),
    epochs = str_replace(model, "spambase_", ""),
    epochs = as.numeric(epochs),
    epsilon = as.character(epsilon),
    mse_ratio = pmax(distance_class_0, distance_class_1) / pmin(distance_class_0, distance_class_1),
    wasserstein_ratio = pmax(wasserstein_class0, wasserstein_class1) / pmin(wasserstein_class0, distance_class_1)
  )



data2 %>%
  pivot_longer(cols = c(wasserstein_ratio, mse_ratio), names_to = "metric", values_to = "ratio") %>%
  ggplot() +
  geom_line(aes(x = epochs, y = ratio, colour = metric)) +
  facet_wrap(vars(epsilon, attack), scales = "free_y")

data2 %>%
  ggplot() +
  geom_line(aes(x = epochs, y = distance_class_1, colour = epsilon)) +
  facet_wrap(~attack)

####################### Boundary Depth Experiment Spambase


data <- read_csv("saved_results/boundary_depth_spambase.csv")


data %>%
  mutate(ratio = pmax(distance_class0, distance_class1) / pmin(distance_class0, distance_class1)) %>%
  ggplot() +
  geom_line(aes(x = hidden_size, y = ratio)) +
  labs(y = "MSE ratio", title = "MSE Ratio by Hidden Layer Size")
