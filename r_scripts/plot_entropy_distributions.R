library(tidyverse)

entropies <- read_csv("saved_results/entropy_distributions.csv")

levels <- c("adversarial", "correct train", "incorrect train", "non-adversarial", "correct test", "incorrect test")

entropies %>%
  mutate(entropy_type = factor(entropy_type, levels = levels)) %>%
  ggplot() +
  geom_density(aes(x = entropy, fill = entropy_type), alpha = 0.7, show.legend = FALSE) +
  coord_cartesian(xlim = c(0, 0.1)) +
  # geom_vline(xintercept = 0.01, colour = "blue") +
  facet_wrap(~entropy_type, scales = "free_y")


entropies %>%
  filter(entropy_type == "adversarial") %>%
  ggplot() +
  geom_density(aes(x = entropy, fill = entropy_type), alpha = 0.7, show.legend = FALSE) +
  geom_vline(xintercept = 0.01, colour = "blue") +
  ggtitle("Adversarial Entropies")

entropies %>%
  group_by(entropy_type) %>%
  summarise(
    min = min(entropy),
    max = max(entropy),
    n = n()
  )

entropies %>%
  group_by(entropy_type) %>%
  summarise(mean(entropy < 0.01))
