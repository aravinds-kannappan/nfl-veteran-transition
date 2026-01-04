# ------------------------------------------------------------------------------
# FILE: R/04_inference_diagnostics.R
# PURPOSE: Visualize Causal Effects, Check Assumptions, Interaction Plots.
# ------------------------------------------------------------------------------

library(ggplot2)
library(sjPlot) # Great for mixed models visualization
library(gridExtra)

model3 <- readRDS("data/models/final_causal_model.rds")
df3 <- readRDS("data/processed/02_modeling_panel.rds")

# ------------------------------------------------------------------------------
# 1. COEFFICIENT FOREST PLOT (Testing Hypothesis 1)
# ------------------------------------------------------------------------------
# Visualizing the interaction effect of Post_Transition * Position

p1 <- plot_model(
  model3,
  type = "int",
  terms = c("post_transition", "position_group")
) +
  theme_minimal() +
  labs(
    title = "Causal Effect of Team Transition by Position",
    y = "Standardized Efficiency (Z-Score)",
    x = "Transition Status"
  ) +
  geom_hline(yintercept = 0, linetype = "dashed")

print(p1)
ggsave("results/plots/interaction_effect.png", p1)

# ------------------------------------------------------------------------------
# 2. RESIDUAL DIAGNOSTICS
# ------------------------------------------------------------------------------
# Check Normality
resids <- resid(model3)
p2 <- ggplot(data.frame(resids), aes(sample = resids)) +
  stat_qq() +
  stat_qq_line() +
  labs(title = "Q-Q Plot of Residuals (LMER)") +
  theme_bw()
print(p2)

# Check Heteroskedasticity (Residuals vs Fitted)
p3 <- ggplot(data.frame(fitted = fitted(model3), resid = resids), aes(x = fitted, y = resid)) +
  geom_point(alpha = 0.3) +
  geom_hline(yintercept = 0, color = "red") +
  geom_smooth(se = FALSE) +
  labs(title = "Residuals vs Fitted") +
  theme_bw()
print(p3)

ggsave("results/plots/diagnostics.png", grid.arrange(p2, p3, ncol=2))

# ------------------------------------------------------------------------------
# 3. REVITALIZATION HYPOTHESIS CHECK
# ------------------------------------------------------------------------------
# Did players with negative pre-trends benefit more?
# Extract Random Slopes
ran_effects <- ranef(model3)$gsis_id
ran_effects$gsis_id <- row.names(ran_effects)

# Merge with Pre-Trend data
check_df <- df3 %>%
  select(gsis_id, pre_trend_slope) %>%
  distinct() %>%
  inner_join(ran_effects, by = "gsis_id")

# Plot
p4 <- ggplot(check_df, aes(x = pre_trend_slope, y = `(Intercept)`)) +
  geom_point() +
  geom_smooth(method = "lm") +
  labs(title = "Pre-Trend vs. Player Intercept",
       x = "Pre-Transition Trend Slope", 
       y = "Player Random Intercept")
print(p4)

ggsave("results/plots/revitalization_check.png", p4)
