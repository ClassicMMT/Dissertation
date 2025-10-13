# This script plots alpha vs. accuracy, lines colour by epochs
# and facetted by:
#     * method (entropy, information_content, probability_gap, combined)
#     * accuracy type (certain, uncertain)


library(tidyverse)
data <- read_csv("saved_results/quantile_accuracy_overfitting_spambase.py") %>%
  pivot_longer(cols = c(certain_accuracy, uncertain_accuracy), values_to = "accuracy", names_to = "accuracy_type") %>%
  filter(
    alpha <= 0.25,
    alpha > 0,
    epoch %in% c(1, 3, 5, 6, 10, 15, 20)
  ) %>%
  mutate(
    accuracy_type = str_remove(accuracy_type, "_accuracy"),
    accuracy = replace_na(accuracy, 0),
    epoch = factor(epoch),
    method = factor(method, levels = c("entropy", "information_content", "probability_gap", "combined"))
  )

data %>%
  ggplot(aes(x = alpha, y = accuracy, colour = epoch)) +
  geom_line() +
  facet_wrap(vars(method, accuracy_type), nrow = 4) +
  theme_minimal() +
  scale_color_brewer(palette = "Set2")

data %>%
  filter(method == "entropy") %>%
  ggplot(aes(x = alpha, y = accuracy, colour = epoch)) +
  geom_line() +
  facet_wrap(vars(accuracy_type), nrow = 1) +
  theme_minimal() +
  scale_color_brewer(palette = "Set2")
