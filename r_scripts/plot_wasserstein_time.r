library(tidyverse)

data <- read_csv("saved_results/wasserstein_time_taken.csv") %>%
  filter(sizes > 50)

x_breaks <- data$sizes

data %>%
  ggplot() +
  geom_col(aes(x = sizes, y = time_taken_seconds), fill = "lightblue", colour = "black") +
  labs(
    x = "Number of Points",
    y = "Time Taken (Seconds)",
    title = "Time Taken To Compute Wasserstein Distance"
  ) +
  theme_minimal() +
  scale_x_continuous(breaks = x_breaks) +
  theme(
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    plot.title = element_text(hjust = 0.5)
  )
