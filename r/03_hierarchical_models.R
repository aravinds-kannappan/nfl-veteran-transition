# ------------------------------------------------------------------------------
# FILE: R/03_hierarchical_models.R
# PURPOSE: Fit Mixed-Effects Models (Random Intercepts/Slopes)
# METHODOLOGY: Models 1 (Null) -> 5 (Heteroskedastic)
# ------------------------------------------------------------------------------

library(lme4)
library(nlme)
library(stargazer) # For regression tables

df2 <- readRDS("data/processed/02_modeling_panel.rds")
df2 <- df2 %>%
  mutate(age_poly1 = age,
         age_poly2 = age^2)

# Ensure factors
df2$gsis_id <- as.factor(df2$gsis_id)
df2$position_group <- as.factor(df2$position_group)

# ------------------------------------------------------------------------------
# MODEL 1: NULL MODEL (Calculating ICC)
# ------------------------------------------------------------------------------
# Determines how much variance is player-specific vs noise
m1_null <- lmer(z_score ~ 1 + (1 | gsis_id), data = df2)
print(summary(m1_null))

# ------------------------------------------------------------------------------
# MODEL 3: FIXED EFFECTS + COVARIATES
# ------------------------------------------------------------------------------
# Controlling for Age (poly), Experience, Team Quality
m3_fixed <- lmer(z_score ~ post_transition + poly(age, 2) + years_exp + 
                   team_quality + pre_trend_slope + 
                   (1 | gsis_id), data = df2)
print(summary(m3_fixed))
# ------------------------------------------------------------------------------
# MODEL 4: RANDOM SLOPES (INDIVIDUAL TRAJECTORIES)
# ------------------------------------------------------------------------------
# Methodology: "Allowing heterogeneous aging and response trajectories"
# We add (1 + rel_time | gsis_id)
m4_random_slopes <- lmer(z_score ~ post_transition + poly(age, 2) + years_exp + 
                           team_quality + 
                           (1 + rel_time | gsis_id), 
                         control = lmerControl(optimizer = "bobyqa"),
                         data = df2)
print(summary(m4_random_slopes))
# ------------------------------------------------------------------------------
# MODEL 5: HETEROSKEDASTIC RESIDUALS (SCHEME FIT TEST)
# ------------------------------------------------------------------------------
# Methodology: Test if variance decreases post-transition using nlme::varIdent
m5_hetero <- lme(
  fixed = z_score ~ post_transition + years_exp,
  random = ~ 1 + rel_time | gsis_id,
  weights = varIdent(form = ~ 1 | post_transition),
  data = df2,
  method = "REML",
  control = lmeControl(
    opt = "optim",
    maxIter = 200
  )
)

print(summary(m5_hetero))
# Likelihood Ratio Test to see if Model 5 improves on Model 4 equivalent
# (Note: Requires refitting m4 in nlme for valid ANOVA)
m4_nlme <- lme(fixed = z_score ~ post_transition + years_exp,
               random = ~ 1 + rel_time | gsis_id,
               data = df2, method = "REML")
print(summary(m4_nlme))
anova_res <- anova(m4_nlme, m5_hetero)
print(anova_res)

# ------------------------------------------------------------------------------
# MODEL 6: POSITION INTERACTIONS (HYPOTHESIS TEST)
# ------------------------------------------------------------------------------
# Hypothesis: delta_RB > delta_WR > delta_QB
m6_interaction <- lmer(z_score ~ post_transition * position_group + 
                         poly(age, 2) + years_exp + team_quality +
                         (1 + rel_time | gsis_id), 
                       data = df2)
print(summary(m6_interaction))
dir.create("data/models", recursive = TRUE, showWarnings = FALSE)
dir.create("results", showWarnings = FALSE)
saveRDS(m6_interaction, "data/models/final_causal_model.rds")
capture.output(summary(m6_interaction), file = "results/model_summary.txt")

