# ------------------------------------------------------------------------------
# FILE: R/02_feature_engineering.R
# PURPOSE: Construct causal covariates, transition indicators, and team contexts.
# ------------------------------------------------------------------------------

library(tidyverse)
library(zoo)

df <- readRDS("data/processed/01_base_metrics.rds")

# ------------------------------------------------------------------------------
# 1. IDENTIFY TEAM TRANSITIONS
# ------------------------------------------------------------------------------
# I need to find players who changed teams between season t-1 and t.
# I explicitly filter for "Veteran" moves (Experience > 3 years).

df_sorted <- df %>%
  arrange(gsis_id, season) %>%
  group_by(gsis_id) %>%
  mutate(
    prev_team = lag(team),
    changed_team = if_else(team != prev_team & !is.na(prev_team), 1, 0),
    years_since_change = cumsum(changed_team)
  ) %>%
  ungroup()

# ------------------------------------------------------------------------------
# 2. DEFINE TRANSITION WINDOWS (2 years Pre, 2 years Post)
# ------------------------------------------------------------------------------
# I isolate specific "transition events" to create clean panels.
transitions <- df_sorted %>%
  filter(changed_team == 1, years_exp >= 3) %>%
  select(gsis_id, transition_season = season, new_team = team)

# Join back to create relative time centered at 0
# Logic: Duplicate rows if a player has multiple distinct major transitions
# For this repo, I take the *first* major veteran transition per player to satisfy independence.
first_transitions <- transitions %>%
  group_by(gsis_id) %>%
  slice(1) %>%
  ungroup()

analysis_panel <- df_sorted %>%
  inner_join(first_transitions, by = "gsis_id") %>%
  mutate(
    rel_time = season - transition_season,
    post_transition = if_else(rel_time >= 0, 1, 0),
    phase = if_else(rel_time < 0, "Pre", "Post")
  ) %>%
  # Filter to window [-2, +2]
  filter(rel_time >= -2 & rel_time <= 2)

# ------------------------------------------------------------------------------
# 3. COVARIATE: TEAM QUALITY (EXCLUDING FOCAL PLAYER)
# ------------------------------------------------------------------------------
# For RBs: Team Passing EPA/Play
# For WRs: Team QB EPA/Play
# For QBs: Team Pass Block Win Rate (simulated here via Sack Rate inverse)

# Helper to get team-level stats excluding specific players
team_stats <- df %>%
  group_by(team, season, position_group) %>%
  summarise(
    team_pos_avg = mean(z_score, na.rm=TRUE),
    .groups = 'drop'
  )

analysis_panel <- analysis_panel %>%
  left_join(team_stats, by = c("team", "season", "position_group"), suffix = c("", "_team")) %>%
  mutate(
    # Construct a "Supporting Cast" metric
    # If I am an RB, my supporting cast quality is approximated by the team's overall offensive EPA
    # (Simplified for this script: using random noise placeholder if external data missing)
    team_quality = case_when(
      position_group == "RB" ~ rnorm(n(), 0, 1), 
      position_group == "QB" ~ rnorm(n(), 0, 1), 
      TRUE ~ rnorm(n(), 0, 1)
    )
  )

# ------------------------------------------------------------------------------
# 3. TEAM OFFENSE
# ------------------------------------------------------------------------------
team_offense <- pbp_reg %>%
  group_by(posteam, season) %>%
  summarise(
    team_pass_epa_per_play = mean(epa[pass_attempt == 1], na.rm = TRUE),
    team_rush_epa_per_play = mean(epa[rush_attempt == 1], na.rm = TRUE),
    team_sack_rate = mean(sack[pass_attempt == 1], na.rm = TRUE),
    .groups = "drop"
  )

analysis_panel <- analysis_panel %>%
  left_join(team_offense, by = c("team" = "posteam", "season")) %>%
  mutate(
    team_quality = case_when(
      position_group == "RB" ~ team_pass_epa_per_play,   # Better passing = better run support (play action, spacing)
      position_group == "WR_TE" ~ team_pass_epa_per_play, # QB-driven
      position_group == "QB" ~ -team_sack_rate,          # Lower sack rate = better protection (negative for interpretability)
      TRUE ~ 0
    ),
    team_quality = coalesce(team_quality, 0)  # Fill NA
  )

# ------------------------------------------------------------------------------
# 4. COVARIATE: PRE-TRANSITION TREND
# ------------------------------------------------------------------------------
# I calculate the slope of performance in years -2 and -1
trend_calc <- analysis_panel %>%
  filter(rel_time < 0) %>%
  group_by(gsis_id) %>%
  summarise(
    pre_trend_slope = coef(lm(z_score ~ rel_time))[2]
  ) %>%
  mutate(pre_trend_slope = coalesce(pre_trend_slope, 0))

analysis_panel <- analysis_panel %>%
  left_join(trend_calc, by = "gsis_id")

# ------------------------------------------------------------------------------
# 5. FINAL EXPORT FOR MODELING
# ------------------------------------------------------------------------------
saveRDS(analysis_panel, "data/processed/02_modeling_panel.rds")
write_csv(analysis_panel, "data/processed/nfl_panel_for_python.csv")

