# ------------------------------------------------------------------------------
# FILE: R/01_data_extraction.R
# AUTHOR: Aravind Kannappan
# PURPOSE: ETL pipeline for nflfastR data (2015-2024), excluding 2020.
#          Calculates standardized metrics (z-scores) for RBs, WRs, QBs.
# ------------------------------------------------------------------------------

library(nflfastR)
library(tidyverse)
library(nflreadr)
library(data.table)
library(here)


# ------------------------------------------------------------------------------
# 1. CONFIGURATION & DATA LOADING
# ------------------------------------------------------------------------------
# Define study years (Skipping 2020 COVID season per methodology)
study_years <- c(2015:2019, 2021:2024)

print("Loading Play-by-Play data... this may take several minutes.")
pbp <- load_pbp(seasons = study_years)

print("Loading Roster data for biographical details...")
rosters <- load_rosters(seasons = study_years)

# ------------------------------------------------------------------------------
# 2. PRE-PROCESSING & FILTERING
# ------------------------------------------------------------------------------
# Filter down to regular season only
pbp_reg <- pbp %>%
  filter(season_type == "REG", !is.na(epa)) %>%
  select(
    play_id, game_id, season, home_team, away_team, posteam, defteam,
    play_type, yards_gained, epa, success, air_yards, yards_after_catch,
    passer_player_id, rusher_player_id, receiver_player_id,
    pass_attempt, rush_attempt, sack, qb_hit
  )

# ------------------------------------------------------------------------------
# 3. METRIC CALCULATION: RUNNING BACKS (YPC)
# ------------------------------------------------------------------------------
rb_stats <- pbp_reg %>%
  filter(rush_attempt == 1, !is.na(rusher_player_id)) %>%
  group_by(rusher_player_id, season) %>%
  summarise(
    metric_raw = mean(yards_gained, na.rm = TRUE),
    volume = n(),
    success_rate = mean(success, na.rm = TRUE),
    total_epa = sum(epa, na.rm = TRUE),
    team = last(posteam),
    .groups = "drop"
  ) %>%
  filter(volume >= 100) %>% # Minimum threshold for stability
  mutate(position_group = "RB") %>%
  rename(gsis_id = rusher_player_id)

# ------------------------------------------------------------------------------
# 4. METRIC CALCULATION: RECEIVERS (YPRR Approximation)
# ------------------------------------------------------------------------------
# Note: True YPRR requires snap counts. We approximate via Yards Per Target 
# multiplied by target share proxies if snap data is unavailable, 
# or use advanced participation data if available in local context.
wr_stats <- pbp_reg %>%
  filter(pass_attempt == 1, !is.na(receiver_player_id)) %>%
  group_by(receiver_player_id, season) %>%
  summarise(
    metric_raw = sum(yards_gained, na.rm = TRUE) / n(), # Yards per Target
    volume = n(),
    success_rate = mean(success, na.rm = TRUE),
    team = last(posteam),
    .groups = "drop"
  ) %>%
  filter(volume >= 50) %>%
  mutate(position_group = "WR_TE") %>%
  rename(gsis_id = receiver_player_id)

# ------------------------------------------------------------------------------
# 5. METRIC CALCULATION: QUARTERBACKS (EPA/Play + CPOE)
# ------------------------------------------------------------------------------
qb_stats <- pbp_reg %>%
  filter((pass_attempt == 1 | rush_attempt == 1), !is.na(passer_player_id)) %>%
  group_by(passer_player_id, season) %>%
  summarise(
    metric_raw = mean(epa, na.rm = TRUE),
    volume = n(),
    metric_secondary = mean(success, na.rm = TRUE), # Success Rate as secondary
    team = last(posteam),
    .groups = "drop"
  ) %>%
  filter(volume >= 200) %>%
  mutate(position_group = "QB") %>%
  rename(gsis_id = passer_player_id)

# ------------------------------------------------------------------------------
# ADD PLAYER AGE
# ------------------------------------------------------------------------------
rosters <- rosters %>%
  mutate(
    birth_date = as.Date(birth_date),
    age = as.numeric(difftime(as.Date(paste(season, "09", "01", sep = "-")), birth_date, units = "days")) / 365.25  # Age as of Sept 1 of season
  ) %>%
  select(gsis_id, season, years_exp, age)

# ------------------------------------------------------------------------------
# 6. MERGING & STANDARDIZATION (Z-SCORES)
# ------------------------------------------------------------------------------
# Combine all positions
all_stats <- bind_rows(rb_stats, wr_stats, qb_stats)

# Calculate Z-Scores within Position-Season to normalize era effects
final_df <- all_stats %>%
  group_by(season, position_group) %>%
  mutate(
    mean_season = mean(metric_raw),
    sd_season = sd(metric_raw),
    z_score = (metric_raw - mean_season) / sd_season
  ) %>%
  ungroup() %>%
  left_join(rosters %>% select(gsis_id, season, years_exp, age), 
            by = c("gsis_id", "season"))

# Save for next step
dir.create(here("data", "processed"), recursive = TRUE, showWarnings = FALSE)
saveRDS(final_df, here("data", "processed", "01_base_metrics.rds"))
print("Data Extraction Complete. File saved to data/processed/01_base_metrics.rds")
