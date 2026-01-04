library(ggplot2)
library(dplyr)
library(sjPlot)
library(gridExtra)
library(ggeffects)  # For marginal effects predictions

model3 <- readRDS("data/models/final_causal_model.rds")
df3 <- readRDS("data/processed/02_modeling_panel.rds")

# Ensure phase for pre/post
df3 <- df3 %>%
  mutate(phase = if_else(post_transition == 1, "Post", "Pre"))

# ------------------------------------------------------------------------------
# 1. PERFORMANCE TRAJECTORY AROUND TRANSITION BY POSITION
# ------------------------------------------------------------------------------
p5 <- ggplot(df3, aes(x = rel_time, y = z_score, color = position_group)) +
  geom_smooth(method = "loess", se = TRUE) +
  geom_vline(xintercept = -0.5, linetype = "dashed", color = "black") +
  labs(title = "Performance Trajectory Pre/Post Transition by Position",
       x = "Relative Time to Transition (0 = Transition Season)",
       y = "Standardized Efficiency (Z-Score)",
       color = "Position") +
  theme_minimal()

print(p5)
ggsave("results/plots/trajectory_by_position.png", p5, width = 10, height = 6)

# ------------------------------------------------------------------------------
# 2. MARGINAL EFFECTS OF TRANSITION BY POSITION (FROM MODEL)
# ------------------------------------------------------------------------------
# Uses ggeffects to compute adjusted predictions
marginal <- ggpredict(model3, terms = c("post_transition", "position_group"))

p6 <- plot(marginal) +
  labs(title = "Model-Estimated Transition Effect by Position",
       y = "Predicted Standardized Efficiency (Z-Score)",
       x = "Transition Status (0 = Pre, 1 = Post)") +
  geom_hline(yintercept = 0, linetype = "dashed") +
  theme_minimal()

print(p6)
ggsave("results/plots/marginal_effects_by_position.png", p6, width = 10, height = 6)

# ------------------------------------------------------------------------------
# 3. TEAM QUALITY CHANGE PRE VS POST BY POSITION
# ------------------------------------------------------------------------------
p7 <- ggplot(df3, aes(x = position_group, y = team_quality, fill = phase)) +
  geom_boxplot() +
  labs(title = "Team Quality (Context) Pre vs. Post Transition by Position",
       x = "Position Group",
       y = "Team Quality Proxy",
       fill = "Phase") +
  theme_minimal()

print(p7)
ggsave("results/plots/team_quality_pre_post_by_pos.png", p7, width = 10, height = 6)

# ------------------------------------------------------------------------------
# 4. REVITALIZATION CHECK: PRE-TREND VS. POST-CHANGE, COLORED BY POSITION
# ------------------------------------------------------------------------------
# First, compute approximate post-transition change (average post - average pre per player)
player_changes <- df3 %>%
  group_by(gsis_id, position_group, pre_trend_slope) %>%
  summarise(
    pre_mean = mean(z_score[rel_time < 0]),
    post_mean = mean(z_score[rel_time >= 0]),
    post_change = post_mean - pre_mean,
    .groups = "drop"
  )

p8 <- ggplot(player_changes, aes(x = pre_trend_slope, y = post_change, color = position_group)) +
  geom_point(alpha = 0.7) +
  geom_smooth(method = "lm", se = FALSE) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  geom_vline(xintercept = 0, linetype = "dashed") +
  labs(title = "Pre-Transition Trend vs. Post-Transition Change by Position",
       subtitle = "Do declining players (negative pre-trend) rebound more?",
       x = "Pre-Transition Performance Slope",
       y = "Post Minus Pre Performance Change (Z-Score)",
       color = "Position") +
  theme_minimal()

print(p8)
ggsave("results/plots/revitalization_by_position.png", p8, width = 10, height = 6)

# Optional: Arrange all new plots in a grid for a summary figure
grid.arrange(p5, p6, p7, p8, ncol = 2)
ggsave("results/plots/additional_visualizations_grid.png", arrangeGrob(p5, p6, p7, p8, ncol = 2), width = 14, height = 10)