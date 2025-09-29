library(tidyverse)

data <- read_csv("saved_results/quantiles_alpha.csv")
# data <- read_csv("saved_results/quantiles_alpha_spambase.csv")

alphas <- data["alpha"]

certain_uncertain <- data %>%
  select(alpha, certain_accuracy, uncertain_accuracy, true_test_accuracy) %>%
  pivot_longer(cols = -alpha, names_to = "accuracy") %>%
  mutate(
    accuracy = str_remove(accuracy, "_accuracy"),
    accuracy = str_replace_all(accuracy, "_", " "),
    accuracy = str_to_title(accuracy),
    accuracy = factor(accuracy, levels = c("Uncertain", "Certain", "True Test"))
  )

certain_uncertain %>%
  filter(alpha <= 0.25) %>%
  ggplot(aes(x = alpha, y = value, colour = accuracy)) +
  geom_line() +
  theme_minimal() +
  labs(y = "Accuracy", colour = "Type", title = "Accuracy by Certain vs. Uncertain Points") +
  theme(panel.grid.minor = element_blank())


false_pos_neg <- data %>%
  select(alpha, false_positive_proportion, false_negative_proportion) %>%
  pivot_longer(cols = -alpha, names_to = "proportion") %>%
  mutate(
    proportion = str_remove(proportion, "_proportion"),
    proportion = str_replace_all(proportion, "_", " "),
    proportion = str_to_title(proportion)
  )

false_pos_neg %>%
  filter(alpha <= 0.25) %>%
  ggplot() +
  geom_line(aes(x = alpha, y = value, colour = proportion)) +
  labs(title = "False Positives + False Negatives", y = "Proportion") +
  scale_y_continuous(breaks = seq(0, 0.15, length.out = 11)) +
  theme_minimal() +
  theme(panel.grid.minor = element_blank())

adversarial_catchment <- data %>%
  select(alpha, proportion_adv_caught_0.01, proportion_adv_caught_0.05) %>%
  pivot_longer(cols = -alpha) %>%
  mutate(name = str_remove(name, "proportion_adv_caught_")) %>%
  rename(proportion_identified = value, epsilon = name)



adversarial_catchment %>%
  filter(alpha <= 0.25) %>%
  ggplot(aes(x = alpha, y = proportion_identified, colour = epsilon)) +
  geom_line() +
  labs(y = "Proportion", title = "Proportion of Adversarial Examples Identified") +
  theme_minimal()
