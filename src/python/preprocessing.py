"""
preprocessing.py
NFL Veterans Team Change Analysis
Purpose: Data cleaning, standardization, and preparation for ML

This script:
1. Loads nfl_panel_full.parquet from enriched data
2. Cleans and standardizes all features
3. Handles missing values intelligently
4. Creates derived metrics
5. Validates data quality
6. Prepares dataset for feature engineering
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Directories
ENRICHED_DIR = Path("data/enriched")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("DATA PREPROCESSING AND STANDARDIZATION")
print("=" * 80)

################################################################################
# 1. LOAD DATA FROM ENRICHED
################################################################################

print("\n1. Loading enriched NFL data...")

try:
    df = pd.read_parquet(ENRICHED_DIR / "nfl_panel_full.parquet")
    print(f"   ✓ Loaded data: {len(df):,} rows × {len(df.columns)} columns")
except FileNotFoundError:
    print("   ❌ Error: Run data_collection.py first")
    exit(1)

################################################################################
# 2. DATA INSPECTION
################################################################################

print("\n2. Inspecting data structure...")

print(f"\n   Columns: {list(df.columns)}")
print(f"\n   Data types:")
print(df.dtypes)

################################################################################
# 3. HANDLE MISSING VALUES
################################################################################

print("\n3. Cleaning and handling missing values...")

missing_report = pd.DataFrame({
    'column': df.columns,
    'missing_count': df.isnull().sum(),
    'missing_pct': (df.isnull().sum() / len(df) * 100).round(2)
})
missing_report = missing_report[missing_report['missing_count'] > 0].sort_values('missing_pct', ascending=False)

if len(missing_report) > 0:
    print(f"   Columns with missing values: {len(missing_report)}")
    print(missing_report.to_string(index=False))
    
    # Strategy: Fill numeric columns with median by group
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            # Fill by position_group if available
            if 'position_group' in df.columns:
                df[col].fillna(df.groupby('position_group')[col].transform('median'), inplace=True)
            # Fill by team if available
            elif 'team' in df.columns:
                df[col].fillna(df.groupby('team')[col].transform('median'), inplace=True)
            # Otherwise use overall median
            else:
                df[col].fillna(df[col].median(), inplace=True)
    
    # Fill categorical columns with mode
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown'
            df[col].fillna(mode_val, inplace=True)
    
    print(f"   ✓ Missing values handled")
else:
    print("   ✓ No missing values detected")

################################################################################
# 4. REMOVE DUPLICATES
################################################################################

print("\n4. Checking for duplicates...")

duplicates = df.duplicated().sum()
print(f"   Duplicate rows found: {duplicates}")

if duplicates > 0:
    df = df.drop_duplicates()
    print(f"   ✓ Removed {duplicates} duplicate rows")

################################################################################
# 5. OUTLIER DETECTION AND HANDLING
################################################################################

print("\n5. Detecting and handling outliers...")

numeric_cols = df.select_dtypes(include=[np.number]).columns
outlier_counts = {}

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 3 * IQR
    upper_bound = Q3 + 3 * IQR
    
    outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
    if outliers > 0:
        outlier_counts[col] = outliers
        # Cap outliers at bounds instead of removing
        df[col] = df[col].clip(lower_bound, upper_bound)

if outlier_counts:
    print(f"   Found and capped outliers in {len(outlier_counts)} columns")
    for col, count in sorted(outlier_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"     • {col}: {count} outliers")
else:
    print("   ✓ No significant outliers detected")

################################################################################
# 6. STANDARDIZATION AND NORMALIZATION
################################################################################

print("\n6. Standardizing numeric features...")

# Select features to standardize (exclude IDs, targets, binary indicators)
exclude_cols = {'gsis_id', 'season', 'position_group', 'team', 'prev_team', 'new_team',
                'changed_team', 'post_transition', 'phase', 'rel_time', 'years_since_change',
                'transition_season', 'performance_level', 'career_phase'}

cols_to_scale = [col for col in numeric_cols if col not in exclude_cols]

# Standardize all numeric features for ML
if len(cols_to_scale) > 0:
    scaler = StandardScaler()
    df_scaled = df[cols_to_scale].fillna(df[cols_to_scale].median())
    df[cols_to_scale] = scaler.fit_transform(df_scaled)
    print(f"   ✓ Standardized {len(cols_to_scale)} numeric features")

################################################################################
# 7. CREATE DERIVED FEATURES
################################################################################

print("\n7. Creating derived features...")

derived_count = 0

# Performance trajectory (trend pre-transition)
if 'pre_trend_slope' in df.columns:
    df['pre_trend_category'] = pd.cut(df['pre_trend_slope'], 
                                       bins=[-np.inf, -0.05, 0.05, np.inf],
                                       labels=['Declining', 'Stable', 'Improving'])
    derived_count += 1

# Team quality interaction
if 'team_quality' in df.columns and 'z_score' in df.columns:
    df['quality_performance_interaction'] = df['team_quality'] * df['z_score']
    derived_count += 1

# Experience groups
if 'years_exp' in df.columns:
    df['exp_group'] = pd.cut(df['years_exp'],
                              bins=[0, 3, 7, 12, np.inf],
                              labels=['Rookie', 'Early Career', 'Prime', 'Veteran'])
    derived_count += 1

# Years since change impact
if 'years_since_change' in df.columns:
    df['adjustment_phase'] = df['years_since_change'].apply(
        lambda x: 'Transition' if x <= 1 else ('Adjustment' if x <= 3 else 'Established')
    )
    derived_count += 1

print(f"   ✓ Created {derived_count} derived features")

################################################################################
# 8. ENCODE CATEGORICAL VARIABLES
################################################################################

print("\n8. Encoding categorical variables...")

categorical_features = df.select_dtypes(include=['object']).columns
encoding_summary = {}

for col in categorical_features:
    # Skip identifiers
    if col in ['gsis_id', 'team', 'prev_team', 'new_team', 'phase']:
        continue
    
    n_unique = df[col].nunique()
    
    if n_unique <= 15:
        # One-hot encode low-cardinality features
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
        df = pd.concat([df, dummies], axis=1)
        encoding_summary[col] = f'one-hot ({n_unique} categories)'
    else:
        # Label encode high-cardinality features
        df[f'{col}_encoded'] = pd.factorize(df[col])[0]
        encoding_summary[col] = f'label-encoded ({n_unique} categories)'

if encoding_summary:
    print(f"   Encoded {len(encoding_summary)} categorical features")
else:
    print("   ✓ No categorical features to encode")

################################################################################
# 9. DATA VALIDATION AND QUALITY CHECKS
################################################################################

print("\n9. Running data quality checks...")

quality_checks = {
    'Total rows': len(df),
    'Total columns': len(df.columns),
    'Duplicate rows': df.duplicated().sum(),
    'Complete cases': (~df.isnull().any(axis=1)).sum(),
    'Completeness %': (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
}

for check, value in quality_checks.items():
    if isinstance(value, float):
        print(f"   ✓ {check}: {value:.2f}%")
    else:
        print(f"   ✓ {check}: {value:,}")

# Data type summary
dtype_summary = df.dtypes.value_counts()
print(f"\n   Data type distribution:")
for dtype, count in dtype_summary.items():
    print(f"     • {dtype}: {count} columns")

################################################################################
# 10. CREATE TRAIN/VALIDATION/TEST INDICATORS
################################################################################

print("\n10. Creating train/validation/test splits...")

# Use relative time if available
if 'rel_time' in df.columns:
    # rel_time likely represents time relative to transition
    # Split by this if it exists
    df['data_split'] = df['rel_time'].apply(
        lambda x: 'train' if x < -2 else ('validation' if x <= 0 else 'test')
    )
    split_counts = df['data_split'].value_counts()
    print(f"   Train: {split_counts.get('train', 0):,} obs (before transition-2)")
    print(f"   Validation: {split_counts.get('validation', 0):,} obs (transition-2 to transition)")
    print(f"   Test: {split_counts.get('test', 0):,} obs (post-transition)")
else:
    # Fallback to random split
    df['data_split'] = np.random.choice(['train', 'validation', 'test'], 
                                         size=len(df), p=[0.7, 0.15, 0.15])
    split_counts = df['data_split'].value_counts()
    print(f"   Train: {split_counts.get('train', 0):,} obs")
    print(f"   Validation: {split_counts.get('validation', 0):,} obs")
    print(f"   Test: {split_counts.get('test', 0):,} obs")

################################################################################
# 11. SAVE PROCESSED DATA
################################################################################

print("\n11. Saving processed data...")

# Save full dataset
df.to_csv(PROCESSED_DIR / "nfl_panel_processed.csv", index=False)
df.to_parquet(PROCESSED_DIR / "nfl_panel_processed.parquet")
print(f"   ✓ Saved: nfl_panel_processed.csv/parquet")

# Save by split
if 'data_split' in df.columns:
    train_df = df[df['data_split'] == 'train']
    val_df = df[df['data_split'] == 'validation']
    test_df = df[df['data_split'] == 'test']
    
    train_df.to_csv(PROCESSED_DIR / "train_dataset.csv", index=False)
    val_df.to_csv(PROCESSED_DIR / "validation_dataset.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test_dataset.csv", index=False)
    
    print(f"   ✓ Saved split datasets (train/val/test)")

# Save preprocessing metadata
metadata = {
    'n_observations': len(df),
    'n_features': len(df.columns),
    'numeric_features': list(cols_to_scale),
    'categorical_features': list(categorical_features),
    'derived_features': derived_count,
    'quality_checks': {k: str(v) for k, v in quality_checks.items()},
    'preprocessing_date': pd.Timestamp.now().isoformat()
}

import json
with open(PROCESSED_DIR / "preprocessing_metadata.json", 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"   ✓ Saved: preprocessing_metadata.json")

################################################################################
# 12. GENERATE PREPROCESSING REPORT
################################################################################

print("\n12. Generating preprocessing report...")

report = f"""
DATA PREPROCESSING REPORT
{'=' * 80}

DATASET OVERVIEW:
- Total observations: {len(df):,}
- Total features: {len(df.columns)}
- Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

DATA QUALITY:
- Completeness: {quality_checks['Completeness %']:.2f}%
- Duplicate rows: {quality_checks['Duplicate rows']}
- Complete cases: {quality_checks['Complete cases']:,}

TRANSFORMATIONS APPLIED:
1. Missing value handling: {len(missing_report)} columns filled
2. Outlier detection: {len(outlier_counts)} columns capped
3. Standardization: {len(cols_to_scale)} numeric features scaled
4. Feature engineering: {len(encoding_summary)} categorical features encoded
5. Derived features: {derived_count} new features created

DATA TYPE DISTRIBUTION:
{dtype_summary.to_string()}

READY FOR FEATURE ENGINEERING AND MODELING
{'=' * 80}
"""

with open(PROCESSED_DIR / "preprocessing_report.txt", 'w') as f:
    f.write(report)

print(report)

################################################################################
# SUMMARY
################################################################################

print("\n" + "=" * 80)
print("DATA PREPROCESSING COMPLETE")
print("=" * 80)
print(f"\nProcessed data saved to: {PROCESSED_DIR}/")
print(f"\nFiles created:")
print(f"  • nfl_panel_processed.csv/parquet")
print(f"  • train_dataset.csv")
print(f"  • validation_dataset.csv")
print(f"  • test_dataset.csv")
print(f"  • preprocessing_metadata.json")
print(f"  • preprocessing_report.txt")
print(f"\nNext step: Run feature_engineering.py")
print("=" * 80 + "\n")

if __name__ == "__main__":
    print("Preprocessing pipeline completed successfully!")
