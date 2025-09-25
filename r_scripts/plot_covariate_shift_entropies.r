library(tidyverse)

data <- read_csv("saved_results/covariate_shift_entropy_exploration.csv")

n_shifted <- 10

data %>%
  filter(features_shifted == n_shifted) %>%
  mutate(is_correct = ifelse(is_correct, "Test Correct", "Test Incorrect")) %>%
  ggplot() +
  geom_density(aes(x = entropy), fill = "lightblue") +
  facet_wrap(vars(is_correct, intensity), scales = "free_y", nrow = 2) +
  theme(
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank(),
    plot.title = element_text(hjust = 0.5),
  ) +
  labs(x = "Entropy", title = paste("Features shifted:", n_shifted))
