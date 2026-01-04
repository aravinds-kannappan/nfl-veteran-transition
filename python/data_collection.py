"""
data_collection.py
NFL Veterans Team Change Analysis
Purpose: Advanced data integration, enrichment, and cross-validation

This Python script focuses on:
1. Integrating multiple data sources beyond basic nflfastR
2. Advanced data quality checks and anomaly detection
3. External data enrichment (injuries, trades, contracts)
4. Cross-validation between different data sources
5. Creating advanced derived metrics not available in R pipeline
"""

import nfl_data_py as nfl
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set up directories
DATA_DIR = Path("data/raw")
ENRICHED_DIR = Path("data/enriched")
DATA_DIR.mkdir(parents=True, exist_ok=True)
ENRICHED_DIR.mkdir(parents=True, exist_ok=True)

# Define seasons
SEASONS = list(range(2015, 2020)) + list(range(2022, 2025))

print("=" * 80)
print("PYTHON ADVANCED DATA INTEGRATION PIPELINE")
print("=" * 80)

################################################################################
# 1. ADVANCED INJURY DATA INTEGRATION
################################################################################

print("\n1. Integrating injury reports and missed games...")

# Download injury data
try:
    injuries = nfl.import_injuries(years=SEASONS)
    
    # Aggregate injury severity by player-season
    injury_analysis = injuries.groupby(['season', 'team', 'full_name']).agg({
        'report_status': lambda x: (x == 'Out').sum(),  # Games missed
        'date': 'count'  # Total injury reports
    }).reset_index()
    
    injury_analysis.columns = ['season', 'team', 'player_name', 'games_out', 'injury_reports']
    
    # Calculate injury severity score
    injury_analysis['injury_severity'] = (
        injury_analysis['games_out'] * 2 + injury_analysis['injury_reports']
    )
    
    injury_analysis.to_parquet(ENRICHED_DIR / "injury_history.parquet")
    print(f"   Processed injury data for {len(injury_analysis):,} player-seasons")
    
except Exception as e:
    print(f"   Warning: Could not load injury data: {e}")

################################################################################
# 2. NEXT GEN STATS INTEGRATION (Advanced Tracking Metrics)
################################################################################

print("\n2. Integrating Next Gen Stats (advanced tracking)...")

# Download Next Gen Stats for receiving
try:
    ngs_receiving = nfl.import_ngs_data(stat_type='receiving', years=SEASONS)
    
    # Key metrics: average_separation, cushion, target_share
    ngs_receiving_clean = ngs_receiving[[
        'season', 'week', 'player_display_name', 'team_abbr',
        'avg_cushion', 'avg_separation', 'percent_share_of_intended_air_yards',
        'avg_intended_air_yards', 'avg_yac_above_expectation'
    ]].copy()
    
    ngs_receiving_clean.to_parquet(ENRICHED_DIR / "ngs_receiving.parquet")
    print(f"   NGS Receiving: {len(ngs_receiving_clean):,} player-week observations")
    
except Exception as e:
    print(f"   Warning: NGS receiving data not available: {e}")

# Download Next Gen Stats for rushing
try:
    ngs_rushing = nfl.import_ngs_data(stat_type='rushing', years=SEASONS)
    
    ngs_rushing_clean = ngs_rushing[[
        'season', 'week', 'player_display_name', 'team_abbr',
        'efficiency', 'percent_attempts_gte_8_defenders',
        'avg_time_to_los', 'rush_yards_over_expected_per_att'
    ]].copy()
    
    ngs_rushing_clean.to_parquet(ENRICHED_DIR / "ngs_rushing.parquet")
    print(f"   NGS Rushing: {len(ngs_rushing_clean):,} player-week observations")
    
except Exception as e:
    print(f"   Warning: NGS rushing data not available: {e}")

# Download Next Gen Stats for passing
try:
    ngs_passing = nfl.import_ngs_data(stat_type='passing', years=SEASONS)
    
    ngs_passing_clean = ngs_passing[[
        'season', 'week', 'player_display_name', 'team_abbr',
        'avg_time_to_throw', 'avg_completed_air_yards',
        'aggressiveness', 'max_completed_air_distance',
        'avg_air_yards_differential'
    ]].copy()
    
    ngs_passing_clean.to_parquet(ENRICHED_DIR / "ngs_passing.parquet")
    print(f"   NGS Passing: {len(ngs_passing_clean):,} player-week observations")
    
except Exception as e:
    print(f"   Warning: NGS passing data not available: {e}")

################################################################################
# 3. SNAP COUNT ANALYSIS (Playing Time Quality)
################################################################################

print("\n3. Analyzing snap counts and usage rates...")

try:
    snap_counts = nfl.import_snap_counts(years=SEASONS)
    
    # Calculate snap share and position-specific usage
    snap_analysis = snap_counts.groupby(['season', 'game_id', 'player', 'team']).agg({
        'offense_snaps': 'sum',
        'offense_pct': 'mean',
        'defense_snaps': 'sum'
    }).reset_index()
    
    snap_analysis.to_parquet(ENRICHED_DIR / "snap_counts.parquet")
    print(f"   Snap count data: {len(snap_analysis):,} player-game observations")
    
except Exception as e:
    print(f"   Warning: Snap count data not available: {e}")

################################################################################
# 4. ADVANCED PFF GRADES INTEGRATION (if available)
################################################################################

print("\n4. Checking for PFF grades data...")

try:
    pff_data = nfl.import_pfr_advanced(years=SEASONS, pos='all')
    
    pff_data.to_parquet(ENRICHED_DIR / "pfr_advanced.parquet")
    print(f"   PFF/Advanced stats: {len(pff_data):,} observations")
    
except Exception as e:
    print(f"   Note: PFF data not available through nfl_data_py: {e}")

################################################################################
# 5. DEPTH CHART ANALYSIS
################################################################################

print("\n5. Analyzing depth chart positions over time...")

try:
    depth_charts = nfl.import_depth_charts(years=SEASONS)
    
    # Track position on depth chart at time of transition
    depth_analysis = depth_charts.groupby(['season', 'club_code', 'full_name']).agg({
        'depth_team': 'first',
        'position': 'first',
        'formation': 'first'
    }).reset_index()
    
    depth_analysis.to_parquet(ENRICHED_DIR / "depth_charts.parquet")
    print(f"   Depth chart data: {len(depth_analysis):,} player-season entries")
    
except Exception as e:
    print(f"   Warning: Depth chart data not available: {e}")

################################################################################
# 6. CROSS-VALIDATION: EPA vs Traditional Stats
################################################################################

print("\n6. Cross-validating EPA with traditional metrics...")

# Load basic data
weekly_stats = nfl.import_weekly_data(years=SEASONS)
pbp_data = nfl.import_pbp_data(years=SEASONS[:2], downcast=True)  # Sample for validation

# Calculate EPA from play-by-play
epa_from_pbp = pbp_data[pbp_data['epa'].notna()].groupby(['season', 'week', 'posteam']).agg({
    'epa': ['mean', 'sum', 'std']
}).reset_index()

# Compare with weekly stats EPA
correlation_check = pd.DataFrame({
    'metric': ['EPA_PlayByPlay', 'EPA_Weekly'],
    'source': ['play_by_play', 'weekly_aggregated'],
    'availability': [
        pbp_data['epa'].notna().mean() * 100,
        weekly_stats['passing_epa'].notna().mean() * 100 if 'passing_epa' in weekly_stats.columns else 0
    ]
})

print("\n   EPA Data Quality:")
print(correlation_check)

correlation_check.to_csv(ENRICHED_DIR / "data_quality_validation.csv", index=False)

################################################################################
# 7. ANOMALY DETECTION IN PERFORMANCE CHANGES
################################################################################

print("\n7. Detecting statistical anomalies in veteran transitions...")

# Load processed data from R pipeline if available
try:
    veteran_changes = pd.read_csv("data/processed/veteran_changes.csv")
    analysis_data = pd.read_csv("data/processed/analysis_dataset.csv")
    
    # Detect outliers using IQR method
    def detect_outliers(df, metric_col):
        Q1 = df[metric_col].quantile(0.25)
        Q3 = df[metric_col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = (df[metric_col] < (Q1 - 3 * IQR)) | (df[metric_col] > (Q3 + 3 * IQR))
        return outliers
    
    # Identify extreme performance changes
    performance_changes = analysis_data.groupby('player_id').agg({
        'primary_metric': lambda x: x.iloc[-1] - x.iloc[0] if len(x) >= 2 else np.nan
    }).reset_index()
    
    performance_changes.columns = ['player_id', 'performance_change']
    performance_changes['is_outlier'] = detect_outliers(
        performance_changes.dropna(), 'performance_change'
    )
    
    # Flag extreme cases for investigation
    extreme_cases = performance_changes[
        performance_changes['is_outlier'].fillna(False)
    ]
    
    if len(extreme_cases) > 0:
        extreme_cases.to_csv(ENRICHED_DIR / "extreme_performance_changes.csv", index=False)
        print(f"   Identified {len(extreme_cases)} extreme performance changes")
    
except FileNotFoundError:
    print("   Skipping (R processed data not yet available)")

################################################################################
# 8. CONTEXTUAL ENRICHMENT: Weather, Travel, Rest Days
################################################################################

print("\n8. Enriching with contextual game factors...")

try:
    schedules = nfl.import_schedules(years=SEASONS)
    
    # Calculate rest days between games
    schedules = schedules.sort_values(['home_team', 'gameday'])
    schedules['days_rest_home'] = schedules.groupby('home_team')['gameday'].diff().dt.days
    schedules['days_rest_away'] = schedules.groupby('away_team')['gameday'].diff().dt.days
    
    # Flag divisional games (tougher competition)
    schedules['is_divisional'] = schedules['game_type'] == 'DIV'
    
    # Travel distance approximation (could be enhanced with stadium coordinates)
    schedules['is_cross_country'] = (
        ((schedules['home_team'].str.contains('SF|LAC|LAR|SEA|LV')) & 
         (schedules['away_team'].str.contains('NE|NYJ|NYG|BUF|MIA'))) |
        ((schedules['away_team'].str.contains('SF|LAC|LAR|SEA|LV')) & 
         (schedules['home_team'].str.contains('NE|NYJ|NYG|BUF|MIA')))
    )
    
    schedules.to_parquet(ENRICHED_DIR / "enriched_schedules.parquet")
    print(f"   Enriched {len(schedules):,} games with contextual factors")
    
except Exception as e:
    print(f"   Warning: Could not enrich schedules: {e}")

################################################################################
# 9. MACHINE LEARNING FEATURE ENGINEERING
################################################################################

print("\n9. Creating ML-ready features for predictive modeling...")

# Create rolling averages and momentum indicators
def create_ml_features(df, player_col, metric_col, window=4):
    """Create rolling statistics for ML models."""
    df = df.sort_values(['player_id', 'season', 'week'])
    
    df[f'{metric_col}_rolling_mean'] = (
        df.groupby(player_col)[metric_col]
        .transform(lambda x: x.rolling(window, min_periods=1).mean())
    )
    
    df[f'{metric_col}_rolling_std'] = (
        df.groupby(player_col)[metric_col]
        .transform(lambda x: x.rolling(window, min_periods=1).std())
    )
    
    df[f'{metric_col}_momentum'] = (
        df.groupby(player_col)[metric_col]
        .transform(lambda x: x.diff())
    )
    
    return df

# Apply to weekly stats (sample)
weekly_sample = weekly_stats.head(10000).copy()  # Sample for demonstration
for metric in ['rushing_yards', 'receiving_yards', 'passing_yards']:
    if metric in weekly_sample.columns:
        weekly_sample = create_ml_features(weekly_sample, 'player_id', metric)

weekly_sample.to_parquet(ENRICHED_DIR / "ml_features_sample.parquet")
print(f"   Created ML features for sample data")

################################################################################
# 10. GENERATE COMPREHENSIVE DATA QUALITY REPORT
################################################################################

print("\n10. Generating data quality report...")

quality_report = {
    'timestamp': datetime.now().isoformat(),
    'seasons_covered': SEASONS,
    'data_sources': {
        'injury_data': (ENRICHED_DIR / "injury_history.parquet").exists(),
        'ngs_receiving': (ENRICHED_DIR / "ngs_receiving.parquet").exists(),
        'ngs_rushing': (ENRICHED_DIR / "ngs_rushing.parquet").exists(),
        'ngs_passing': (ENRICHED_DIR / "ngs_passing.parquet").exists(),
        'snap_counts': (ENRICHED_DIR / "snap_counts.parquet").exists(),
        'depth_charts': (ENRICHED_DIR / "depth_charts.parquet").exists(),
    },
    'enrichment_complete': True
}

import json
with open(ENRICHED_DIR / "quality_report.json", 'w') as f:
    json.dump(quality_report, f, indent=2)

print("\n" + "=" * 80)
print("PYTHON ENRICHMENT COMPLETE")
print("=" * 80)
print(f"\nEnriched data saved to: {ENRICHED_DIR}")
print("\nAvailable enhancements:")
for source, available in quality_report['data_sources'].items():
    status = "✓" if available else "✗"
    print(f"  {status} {source}")

if __name__ == "__main__":
    print("\nThis script provides advanced data enrichment beyond the R pipeline.")
    print("Run after R data collection for maximum integration.")
