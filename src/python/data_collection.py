"""
data_collection.py
NFL Veterans Team Change Analysis
Purpose: Data loading, validation, and initial enrichment

This script focuses on:
1. Loading nfl_panel_for_python.csv
2. Data validation and quality checks
3. Creating basic derived metrics
4. Saving to enriched format
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set up directories
DATA_DIR = Path("data")
ENRICHED_DIR = Path("data/enriched")
DATA_DIR.mkdir(parents=True, exist_ok=True)
ENRICHED_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("PYTHON DATA LOADING AND VALIDATION PIPELINE")
print("=" * 80)

################################################################################
# 1. LOAD DATA FROM CSV
################################################################################

print("\n1. Loading nfl_panel_for_python.csv...")

try:
    df = pd.read_csv(DATA_DIR / "nfl_panel_for_python.csv")
    print(f"   ✓ Successfully loaded: {len(df):,} rows × {len(df.columns)} columns")
except FileNotFoundError:
    print(f"   ❌ Error: nfl_panel_for_python.csv not found in {DATA_DIR}/")
    exit(1)

################################################################################
# 2. DATA STRUCTURE INSPECTION
################################################################################

print("\n2. Inspecting data structure...")

print(f"\n   Column names and types:")
for col in df.columns:
    print(f"     • {col}: {df[col].dtype}")

print(f"\n   Data shape: {df.shape}")
print(f"\n   First 5 rows:")
print(df.head())

################################################################################
# 3. BASIC DATA QUALITY CHECKS
################################################################################

print("\n3. Running data quality checks...")

quality_metrics = {
    'Total rows': len(df),
    'Total columns': len(df.columns),
    'Memory usage (MB)': df.memory_usage(deep=True).sum() / 1024**2,
    'Duplicate rows': df.duplicated().sum(),
    'Rows with any nulls': df.isnull().any(axis=1).sum(),
}

for metric, value in quality_metrics.items():
    if isinstance(value, float):
        print(f"   ✓ {metric}: {value:.2f}")
    else:
        print(f"   ✓ {metric}: {value:,}")

# Missing value analysis
print("\n   Missing value analysis:")
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)

if len(missing) > 0:
    for col, count in missing.items():
        pct = count / len(df) * 100
        print(f"     • {col}: {count:,} ({pct:.1f}%)")
else:
    print("     ✓ No missing values detected")

################################################################################
# 4. DESCRIPTIVE STATISTICS
################################################################################

print("\n4. Descriptive statistics...")

print("\n   Numeric columns summary:")
print(df.describe())

print("\n   Categorical columns:")
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    print(f"     • {col}: {df[col].nunique()} unique values")

################################################################################
# 5. KEY VARIABLES VALIDATION
################################################################################

print("\n5. Validating key variables...")

# Check season range
if 'season' in df.columns:
    seasons = sorted(df['season'].unique())
    print(f"   Seasons: {seasons[0]} to {seasons[-1]}")

# Check positions
if 'position_group' in df.columns:
    positions = df['position_group'].value_counts()
    print(f"   Position groups:")
    for pos, count in positions.items():
        print(f"     • {pos}: {count}")

# Check transition indicator
if 'post_transition' in df.columns:
    transition_counts = df['post_transition'].value_counts()
    print(f"   Post-transition indicator:")
    for val, count in transition_counts.items():
        print(f"     • {val}: {count}")

# Check team changes
if 'changed_team' in df.columns:
    changes = df['changed_team'].value_counts()
    print(f"   Team changes:")
    for val, count in changes.items():
        label = "Yes" if val == 1 else "No"
        print(f"     • Changed ({label}): {count}")

################################################################################
# 6. CREATE DERIVED METRICS
################################################################################

print("\n6. Creating derived metrics...")

# Already have z_score, but create additional standardized versions
if 'metric_raw' in df.columns and 'sd_season' in df.columns:
    df['metric_raw_scaled'] = df['metric_raw'] / (df['sd_season'] + 1e-10)
    print(f"   ✓ Created scaled metric")

# Create performance level categories
if 'z_score' in df.columns:
    df['performance_level'] = pd.cut(df['z_score'], 
                                      bins=[-np.inf, -1, 0, 1, np.inf],
                                      labels=['Below Avg', 'Average', 'Above Avg', 'Elite'])
    print(f"   ✓ Created performance level categories")

# Create career phase
if 'years_exp' in df.columns:
    df['career_phase'] = pd.cut(df['years_exp'],
                                bins=[0, 3, 7, 12, np.inf],
                                labels=['Early', 'Prime', 'Veteran', 'Late Career'])
    print(f"   ✓ Created career phase categories")

# Age squared for polynomial terms
if 'age' in df.columns:
    df['age_squared'] = df['age'] ** 2
    print(f"   ✓ Created age polynomial features")

################################################################################
# 7. PLAYER-SEASON SUMMARIES
################################################################################

print("\n7. Creating player-season summaries...")

if 'gsis_id' in df.columns and 'season' in df.columns:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    player_season_stats = df.groupby(['gsis_id', 'season'])[numeric_cols].agg([
        'mean', 'sum', 'std', 'count'
    ]).reset_index()
    
    player_season_stats.to_parquet(ENRICHED_DIR / "player_season_summary.parquet")
    print(f"   ✓ Saved player-season summaries: {len(player_season_stats)} records")

################################################################################
# 8. TEAM-SEASON SUMMARIES
################################################################################

print("\n8. Creating team-season summaries...")

if 'team' in df.columns and 'season' in df.columns:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    team_season_stats = df.groupby(['team', 'season'])[numeric_cols].agg([
        'mean', 'sum', 'std'
    ]).reset_index()
    
    team_season_stats.to_parquet(ENRICHED_DIR / "team_season_summary.parquet")
    print(f"   ✓ Saved team-season summaries: {len(team_season_stats)} records")

################################################################################
# 9. TRANSITION ANALYSIS
################################################################################

print("\n9. Analyzing transitions...")

if 'changed_team' in df.columns and 'gsis_id' in df.columns:
    transitions = df[df['changed_team'] == 1][['gsis_id', 'season', 'position_group', 
                                                 'prev_team', 'new_team', 'metric_raw', 'z_score']]
    transitions = transitions.sort_values(['gsis_id', 'season'])
    
    if len(transitions) > 0:
        transitions.to_parquet(ENRICHED_DIR / "team_transitions.parquet")
        print(f"   ✓ Identified {len(transitions)} team transition events")
        
        # Summary statistics
        print(f"\n   Transition summary:")
        print(f"     • Unique players: {transitions['gsis_id'].nunique()}")
        print(f"     • Positions affected:")
        for pos, count in transitions['position_group'].value_counts().items():
            print(f"       - {pos}: {count}")

################################################################################
# 10. SAVE ENRICHED DATASET
################################################################################

print("\n10. Saving enriched dataset...")

# Save full dataset in multiple formats
df.to_parquet(ENRICHED_DIR / "nfl_panel_full.parquet")
df.to_csv(ENRICHED_DIR / "nfl_panel_full.csv", index=False)

print(f"   ✓ Saved: nfl_panel_full.parquet")
print(f"   ✓ Saved: nfl_panel_full.csv")

################################################################################
# 11. GENERATE DATA QUALITY REPORT
################################################################################

print("\n11. Generating data quality report...")

quality_report = {
    'timestamp': datetime.now().isoformat(),
    'total_rows': len(df),
    'total_columns': len(df.columns),
    'columns': list(df.columns),
    'memory_usage_mb': float(df.memory_usage(deep=True).sum() / 1024**2),
    'duplicate_rows': int(df.duplicated().sum()),
    'missing_values': df.isnull().sum().to_dict(),
}

import json
with open(ENRICHED_DIR / "data_quality_report.json", 'w') as f:
    json.dump(quality_report, f, indent=2, default=str)

print(f"   ✓ Saved: data_quality_report.json")

################################################################################
# SUMMARY
################################################################################

print("\n" + "=" * 80)
print("DATA LOADING COMPLETE")
print("=" * 80)
print(f"\nData summary:")
print(f"  • Records: {len(df):,}")
print(f"  • Features: {len(df.columns)}")
print(f"  • Seasons: {df['season'].min() if 'season' in df.columns else 'N/A'} to {df['season'].max() if 'season' in df.columns else 'N/A'}")

print(f"\nEnriched data saved to: {ENRICHED_DIR}/")
print(f"\nFiles created:")
print(f"  ✓ nfl_panel_full.parquet")
print(f"  ✓ nfl_panel_full.csv")
print(f"  ✓ player_season_summary.parquet")
print(f"  ✓ team_season_summary.parquet")
print(f"  ✓ team_transitions.parquet")
print(f"  ✓ data_quality_report.json")

print(f"\nNext step: Run preprocessing.py")
print("=" * 80 + "\n")

if __name__ == "__main__":
    print("Data loading pipeline completed!")
