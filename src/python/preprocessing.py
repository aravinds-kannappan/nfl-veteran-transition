"""
preprocessing.py
NFL Veterans Team Change Analysis
Purpose: Data cleaning, standardization, and preparation for ML

This script:
1. Loads raw data from R pipeline and enriched Python data
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
RAW_DIR = Path("data/raw")
ENRICHED_DIR = Path("data/enriched")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("DATA PREPROCESSING AND STANDARDIZATION")
print("=" * 80)

################################################################################
# 1. LOAD DATA FROM R PIPELINE AND PYTHON ENRICHMENT
################################################################################

print("\n1. Loading data from previous pipelines...")

# Load R-processed data
try:
    analysis_dataset = pd.read_csv(PROCESSED_DIR / "analysis_dataset.csv")
    veteran_changes = pd.read_csv(PROCESSED_DIR / "veteran_changes.csv")
    print(f"   ✓ Loaded R analysis dataset: {len(analysis_dataset)} rows")
    print(f"   ✓ Loaded veteran changes: {len(veteran_changes)} rows")
except FileNotFoundError:
    print("   ⚠ R processed data not found - will create minimal dataset")
    analysis_dataset = pd.DataFrame()
    veteran_changes = pd.DataFrame()

# Load enriched data from Python pipeline
try:
    injury_data = pd.read_parquet(ENRICHED_DIR / "injury_history.parquet")
    print(f"   ✓ Loaded injury data: {len(injury_data)} rows")
except FileNotFoundError:
    print("   ⚠ Injury data not available")
    injury_data = None

try:
    snap_counts = pd.read_parquet(ENRICHED_DIR / "snap_counts.parquet")
    print(f"   ✓ Loaded snap count data: {len(snap_counts)} rows")
except FileNotFoundError:
    print("   ⚠ Snap count data not available")
    snap_counts = None

try:
    schedules = pd.read_parquet(ENRICHED_DIR / "enriched_schedules.parquet")
    print(f"   ✓ Loaded enriched schedules: {len(schedules)} rows")
except FileNotFoundError:
    print("   ⚠ Schedule data not available")
    schedules = None

################################################################################
# 2. DATA CLEANING - HANDLE MISSING VALUES
################################################################################

print("\n2. Cleaning and handling missing values...")

# Work with analysis dataset
df = analysis_dataset.copy()

if len(df) > 0:
    # Identify missing value patterns
    missing_report = pd.DataFrame({
        'column': df.columns,
        'missing_count': df.isnull().sum(),
        'missing_pct': (df.isnull().sum() / len(df) * 100).round(2)
    })
    missing_report = missing_report[missing_report['missing_count'] > 0].sort_values('missing_pct', ascending=False)
    
    print(f"   Columns with missing values: {len(missing_report)}")
    if len(missing_report) > 0:
        print(missing_report.to_string(index=False))
    
    # Strategy: Fill numeric columns with median by group
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            # Fill by position if available
            if 'position' in df.columns:
                df[col].fillna(df.groupby('position')[col].transform('median'), inplace=True)
            else:
                df[col].fillna(df[col].median(), inplace=True)
    
    # Fill categorical columns with mode
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown', inplace=True)
    
    print(f"   ✓ Missing values handled")

################################################################################
# 3. OUTLIER DETECTION AND HANDLING
################################################################################

print("\n3. Detecting and handling outliers...")

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
    print("   No significant outliers detected")

################################################################################
# 4. STANDARDIZATION AND NORMALIZATION
################################################################################

print("\n4. Standardizing numeric features...")

# Select features to standardize (exclude IDs, targets, binary indicators)
exclude_cols = {'player_id', 'season', 'week', 'game_id', 'transition_year', 
                'relative_year', 'post', 'player_name', 'team', 'position', 'recent_team'}
cols_to_scale = [col for col in numeric_cols if col not in exclude_cols]

# Create z-scored versions of key metrics
primary_metric_cols = [col for col in cols_to_scale if 'metric' in col.lower()]

for col in primary_metric_cols:
    if col in df.columns:
        df[f'{col}_z'] = (df[col] - df[col].mean()) / (df[col].std() + 1e-8)

print(f"   ✓ Created z-scored versions of {len(primary_metric_cols)} metrics")

# Standardize all numeric features
scaler = StandardScaler()
df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale].fillna(0))

print(f"   ✓ Standardized {len(cols_to_scale)} numeric features")

################################################################################
# 5. CREATE DERIVED FEATURES
################################################################################

print("\n5. Creating derived features...")

# Age-squared for non-linear aging effects
if 'age' in df.columns:
    df['age_squared'] = df['age'] ** 2
    df['age_cubed'] = df['age'] ** 3

# Experience-related features
if 'experience' in df.columns:
    df['experience_squared'] = df['experience'] ** 2
    df['career_stage'] = pd.cut(df['experience'], bins=[0, 3, 7, 10, 20], 
                                 labels=['Early', 'Prime', 'Veteran', 'Twilight'])

# Performance trend (year-over-year change)
if 'primary_metric_z' in df.columns and 'player_id' in df.columns:
    df = df.sort_values(['player_id', 'season'])
    df['yoy_change'] = df.groupby('player_id')['primary_metric_z'].diff()
    df['yoy_pct_change'] = df.groupby('player_id')['primary_metric_z'].pct_change()

# Injury impact (if available)
if injury_data is not None and len(injury_data) > 0:
    injury_summary = injury_data.groupby(['season', 'player_name']).agg({
        'games_out': 'sum',
        'injury_severity': 'max'
    }).reset_index()
    
    df = df.merge(injury_summary, 
                  left_on=['season', 'player_name'], 
                  right_on=['season', 'player_name'],
                  how='left')
    df['games_out'].fillna(0, inplace=True)
    df['injury_severity'].fillna(0, inplace=True)
    print(f"   ✓ Integrated injury data")

# Snap count integration (if available)
if snap_counts is not None and len(snap_counts) > 0:
    snap_summary = snap_counts.groupby(['season', 'player']).agg({
        'offense_snaps': 'mean',
        'offense_pct': 'mean'
    }).reset_index()
    snap_summary.columns = ['season', 'player_name', 'avg_snaps', 'avg_snap_pct']
    
    df = df.merge(snap_summary,
                  left_on=['season', 'player_name'],
                  right_on=['season', 'player_name'],
                  how='left')
    print(f"   ✓ Integrated snap count data")

print(f"   ✓ Created derived features (total columns: {len(df.columns)})")

################################################################################
# 6. ENCODE CATEGORICAL VARIABLES
################################################################################

print("\n6. Encoding categorical variables...")

categorical_features = df.select_dtypes(include=['object']).columns
encoding_summary = {}

for col in categorical_features:
    if col not in ['player_name', 'team', 'recent_team']:  # Keep identifiers
        n_unique = df[col].nunique()
        
        if n_unique <= 10:
            # One-hot encode low-cardinality features
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df, dummies], axis=1)
            encoding_summary[col] = f'one-hot ({n_unique} categories)'
        else:
            # Label encode high-cardinality features
            df[f'{col}_encoded'] = df[col].astype('category').cat.codes
            encoding_summary[col] = f'label-encoded ({n_unique} categories)'

print(f"   Encoded {len(encoding_summary)} categorical features:")
for feat, method in list(encoding_summary.items())[:5]:
    print(f"     • {feat}: {method}")
if len(encoding_summary) > 5:
    print(f"     ... and {len(encoding_summary) - 5} more")

################################################################################
# 7. DATA VALIDATION AND QUALITY CHECKS
################################################################################

print("\n7. Running data quality checks...")

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
# 8. CREATE TRAIN/VALIDATION/TEST INDICATORS
################################################################################

print("\n8. Creating train/validation/test splits...")

# Temporal split (by season for time-series validation)
if 'season' in df.columns:
    unique_seasons = sorted(df['season'].unique())
    n_seasons = len(unique_seasons)
    
    train_threshold = unique_seasons[int(n_seasons * 0.7)]
    val_threshold = unique_seasons[int(n_seasons * 0.85)]
    
    df['data_split'] = 'train'
    df.loc[df['season'] > val_threshold, 'data_split'] = 'test'
    df.loc[(df['season'] > train_threshold) & (df['season'] <= val_threshold), 'data_split'] = 'validation'
    
    split_counts = df['data_split'].value_counts()
    print(f"   Train: {split_counts.get('train', 0):,} obs ({train_threshold})")
    print(f"   Validation: {split_counts.get('validation', 0):,} obs ({train_threshold}-{val_threshold})")
    print(f"   Test: {split_counts.get('test', 0):,} obs (>{val_threshold})")

################################################################################
# 9. SAVE PROCESSED DATA
################################################################################

print("\n9. Saving processed data...")

# Save full dataset
df.to_csv(PROCESSED_DIR / "analysis_dataset_processed.csv", index=False)
df.to_parquet(PROCESSED_DIR / "analysis_dataset_processed.parquet")
print(f"   ✓ Saved: analysis_dataset_processed.csv/parquet")

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
    'numeric_features': cols_to_scale,
    'categorical_features': list(categorical_features),
    'derived_features': list(encoding_summary.keys()),
    'quality_checks': {k: str(v) for k, v in quality_checks.items()},
    'preprocessing_date': pd.Timestamp.now().isoformat()
}

import json
with open(PROCESSED_DIR / "preprocessing_metadata.json", 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"   ✓ Saved: preprocessing_metadata.json")

################################################################################
# 10. GENERATE PREPROCESSING REPORT
################################################################################

print("\n10. Generating preprocessing report...")

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
5. Derived features: Age polynomials, career trajectory, YoY changes

DATA TYPE DISTRIBUTION:
{dtype_summary.to_string()}

SPLITS:
- Training: {split_counts.get('train', 0):,} observations
- Validation: {split_counts.get('validation', 0):,} observations
- Testing: {split_counts.get('test', 0):,} observations

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
print(f"\nProcessed data saved to: {PROCESSED_DIR}")
print(f"\nFiles created:")
print(f"  • analysis_dataset_processed.csv/parquet")
print(f"  • train_dataset.csv")
print(f"  • validation_dataset.csv")
print(f"  • test_dataset.csv")
print(f"  • preprocessing_metadata.json")
print(f"  • preprocessing_report.txt")
print(f"\nNext step: Run feature_engineering.py")
print("=" * 80 + "\n")

if __name__ == "__main__":
    print("Preprocessing pipeline completed successfully!")"""
preprocessing.py
NFL Veterans Team Change Analysis
Purpose: Data cleaning, standardization, and preparation for ML

This script:
1. Loads raw data from R pipeline and enriched Python data
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
RAW_DIR = Path("data/raw")
ENRICHED_DIR = Path("data/enriched")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("DATA PREPROCESSING AND STANDARDIZATION")
print("=" * 80)

################################################################################
# 1. LOAD DATA FROM R PIPELINE AND PYTHON ENRICHMENT
################################################################################

print("\n1. Loading data from previous pipelines...")

# Load R-processed data
try:
    analysis_dataset = pd.read_csv(PROCESSED_DIR / "analysis_dataset.csv")
    veteran_changes = pd.read_csv(PROCESSED_DIR / "veteran_changes.csv")
    print(f"   ✓ Loaded R analysis dataset: {len(analysis_dataset)} rows")
    print(f"   ✓ Loaded veteran changes: {len(veteran_changes)} rows")
except FileNotFoundError:
    print("   ⚠ R processed data not found - will create minimal dataset")
    analysis_dataset = pd.DataFrame()
    veteran_changes = pd.DataFrame()

# Load enriched data from Python pipeline
try:
    injury_data = pd.read_parquet(ENRICHED_DIR / "injury_history.parquet")
    print(f"   ✓ Loaded injury data: {len(injury_data)} rows")
except FileNotFoundError:
    print("   ⚠ Injury data not available")
    injury_data = None

try:
    snap_counts = pd.read_parquet(ENRICHED_DIR / "snap_counts.parquet")
    print(f"   ✓ Loaded snap count data: {len(snap_counts)} rows")
except FileNotFoundError:
    print("   ⚠ Snap count data not available")
    snap_counts = None

try:
    schedules = pd.read_parquet(ENRICHED_DIR / "enriched_schedules.parquet")
    print(f"   ✓ Loaded enriched schedules: {len(schedules)} rows")
except FileNotFoundError:
    print("   ⚠ Schedule data not available")
    schedules = None

################################################################################
# 2. DATA CLEANING - HANDLE MISSING VALUES
################################################################################

print("\n2. Cleaning and handling missing values...")

# Work with analysis dataset
df = analysis_dataset.copy()

if len(df) > 0:
    # Identify missing value patterns
    missing_report = pd.DataFrame({
        'column': df.columns,
        'missing_count': df.isnull().sum(),
        'missing_pct': (df.isnull().sum() / len(df) * 100).round(2)
    })
    missing_report = missing_report[missing_report['missing_count'] > 0].sort_values('missing_pct', ascending=False)
    
    print(f"   Columns with missing values: {len(missing_report)}")
    if len(missing_report) > 0:
        print(missing_report.to_string(index=False))
    
    # Strategy: Fill numeric columns with median by group
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            # Fill by position if available
            if 'position' in df.columns:
                df[col].fillna(df.groupby('position')[col].transform('median'), inplace=True)
            else:
                df[col].fillna(df[col].median(), inplace=True)
    
    # Fill categorical columns with mode
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown', inplace=True)
    
    print(f"   ✓ Missing values handled")

################################################################################
# 3. OUTLIER DETECTION AND HANDLING
################################################################################

print("\n3. Detecting and handling outliers...")

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
    print("   No significant outliers detected")

################################################################################
# 4. STANDARDIZATION AND NORMALIZATION
################################################################################

print("\n4. Standardizing numeric features...")

# Select features to standardize (exclude IDs, targets, binary indicators)
exclude_cols = {'player_id', 'season', 'week', 'game_id', 'transition_year', 
                'relative_year', 'post', 'player_name', 'team', 'position', 'recent_team'}
cols_to_scale = [col for col in numeric_cols if col not in exclude_cols]

# Create z-scored versions of key metrics
primary_metric_cols = [col for col in cols_to_scale if 'metric' in col.lower()]

for col in primary_metric_cols:
    if col in df.columns:
        df[f'{col}_z'] = (df[col] - df[col].mean()) / (df[col].std() + 1e-8)

print(f"   ✓ Created z-scored versions of {len(primary_metric_cols)} metrics")

# Standardize all numeric features
scaler = StandardScaler()
df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale].fillna(0))

print(f"   ✓ Standardized {len(cols_to_scale)} numeric features")

################################################################################
# 5. CREATE DERIVED FEATURES
################################################################################

print("\n5. Creating derived features...")

# Age-squared for non-linear aging effects
if 'age' in df.columns:
    df['age_squared'] = df['age'] ** 2
    df['age_cubed'] = df['age'] ** 3

# Experience-related features
if 'experience' in df.columns:
    df['experience_squared'] = df['experience'] ** 2
    df['career_stage'] = pd.cut(df['experience'], bins=[0, 3, 7, 10, 20], 
                                 labels=['Early', 'Prime', 'Veteran', 'Twilight'])

# Performance trend (year-over-year change)
if 'primary_metric_z' in df.columns and 'player_id' in df.columns:
    df = df.sort_values(['player_id', 'season'])
    df['yoy_change'] = df.groupby('player_id')['primary_metric_z'].diff()
    df['yoy_pct_change'] = df.groupby('player_id')['primary_metric_z'].pct_change()

# Injury impact (if available)
if injury_data is not None and len(injury_data) > 0:
    injury_summary = injury_data.groupby(['season', 'player_name']).agg({
        'games_out': 'sum',
        'injury_severity': 'max'
    }).reset_index()
    
    df = df.merge(injury_summary, 
                  left_on=['season', 'player_name'], 
                  right_on=['season', 'player_name'],
                  how='left')
    df['games_out'].fillna(0, inplace=True)
    df['injury_severity'].fillna(0, inplace=True)
    print(f"   ✓ Integrated injury data")

# Snap count integration (if available)
if snap_counts is not None and len(snap_counts) > 0:
    snap_summary = snap_counts.groupby(['season', 'player']).agg({
        'offense_snaps': 'mean',
        'offense_pct': 'mean'
    }).reset_index()
    snap_summary.columns = ['season', 'player_name', 'avg_snaps', 'avg_snap_pct']
    
    df = df.merge(snap_summary,
                  left_on=['season', 'player_name'],
                  right_on=['season', 'player_name'],
                  how='left')
    print(f"   ✓ Integrated snap count data")

print(f"   ✓ Created derived features (total columns: {len(df.columns)})")

################################################################################
# 6. ENCODE CATEGORICAL VARIABLES
################################################################################

print("\n6. Encoding categorical variables...")

categorical_features = df.select_dtypes(include=['object']).columns
encoding_summary = {}

for col in categorical_features:
    if col not in ['player_name', 'team', 'recent_team']:  # Keep identifiers
        n_unique = df[col].nunique()
        
        if n_unique <= 10:
            # One-hot encode low-cardinality features
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df, dummies], axis=1)
            encoding_summary[col] = f'one-hot ({n_unique} categories)'
        else:
            # Label encode high-cardinality features
            df[f'{col}_encoded'] = df[col].astype('category').cat.codes
            encoding_summary[col] = f'label-encoded ({n_unique} categories)'

print(f"   Encoded {len(encoding_summary)} categorical features:")
for feat, method in list(encoding_summary.items())[:5]:
    print(f"     • {feat}: {method}")
if len(encoding_summary) > 5:
    print(f"     ... and {len(encoding_summary) - 5} more")

################################################################################
# 7. DATA VALIDATION AND QUALITY CHECKS
################################################################################

print("\n7. Running data quality checks...")

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
# 8. CREATE TRAIN/VALIDATION/TEST INDICATORS
################################################################################

print("\n8. Creating train/validation/test splits...")

# Temporal split (by season for time-series validation)
if 'season' in df.columns:
    unique_seasons = sorted(df['season'].unique())
    n_seasons = len(unique_seasons)
    
    train_threshold = unique_seasons[int(n_seasons * 0.7)]
    val_threshold = unique_seasons[int(n_seasons * 0.85)]
    
    df['data_split'] = 'train'
    df.loc[df['season'] > val_threshold, 'data_split'] = 'test'
    df.loc[(df['season'] > train_threshold) & (df['season'] <= val_threshold), 'data_split'] = 'validation'
    
    split_counts = df['data_split'].value_counts()
    print(f"   Train: {split_counts.get('train', 0):,} obs ({train_threshold})")
    print(f"   Validation: {split_counts.get('validation', 0):,} obs ({train_threshold}-{val_threshold})")
    print(f"   Test: {split_counts.get('test', 0):,} obs (>{val_threshold})")

################################################################################
# 9. SAVE PROCESSED DATA
################################################################################

print("\n9. Saving processed data...")

# Save full dataset
df.to_csv(PROCESSED_DIR / "analysis_dataset_processed.csv", index=False)
df.to_parquet(PROCESSED_DIR / "analysis_dataset_processed.parquet")
print(f"   ✓ Saved: analysis_dataset_processed.csv/parquet")

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
    'numeric_features': cols_to_scale,
    'categorical_features': list(categorical_features),
    'derived_features': list(encoding_summary.keys()),
    'quality_checks': {k: str(v) for k, v in quality_checks.items()},
    'preprocessing_date': pd.Timestamp.now().isoformat()
}

import json
with open(PROCESSED_DIR / "preprocessing_metadata.json", 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"   ✓ Saved: preprocessing_metadata.json")

################################################################################
# 10. GENERATE PREPROCESSING REPORT
################################################################################

print("\n10. Generating preprocessing report...")

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
5. Derived features: Age polynomials, career trajectory, YoY changes

DATA TYPE DISTRIBUTION:
{dtype_summary.to_string()}

SPLITS:
- Training: {split_counts.get('train', 0):,} observations
- Validation: {split_counts.get('validation', 0):,} observations
- Testing: {split_counts.get('test', 0):,} observations

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
print(f"\nProcessed data saved to: {PROCESSED_DIR}")
print(f"\nFiles created:")
print(f"  • analysis_dataset_processed.csv/parquet")
print(f"  • train_dataset.csv")
print(f"  • validation_dataset.csv")
print(f"  • test_dataset.csv")
print(f"  • preprocessing_metadata.json")
print(f"  • preprocessing_report.txt")
print(f"\nNext step: Run feature_engineering.py")
print("=" * 80 + "\n")

if __name__ == "__main__":
    print("Preprocessing pipeline completed successfully!")
