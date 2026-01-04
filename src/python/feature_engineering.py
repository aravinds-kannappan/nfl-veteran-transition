"""
feature_engineering.py
NFL Veterans Team Change Analysis  
Purpose: ML-focused feature engineering complementing R statistical analysis

Creates features for:
- Predicting post-transition success
- Propensity score matching
- Career trajectory forecasting  
- Causal effect heterogeneity
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from scipy.stats import zscore
import warnings
warnings.filterwarnings('ignore')

# Directories
PROCESSED_DIR = Path("data/processed")
ENRICHED_DIR = Path("data/enriched")
ML_DIR = Path("data/ml_features")
ML_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("MACHINE LEARNING FEATURE ENGINEERING")
print("Complementing R statistical inference with ML prediction")
print("=" * 80)

# Load data from R pipeline
try:
    analysis_data = pd.read_csv(PROCESSED_DIR / "analysis_dataset.csv")
    print(f"\nLoaded {len(analysis_data)} observations from R pipeline")
except FileNotFoundError:
    print("Error: Run R preprocessing first (02_preprocessing.R)")
    exit(1)

################################################################################
# 1. PROPENSITY SCORE FEATURES (for matching/weighting)
################################################################################

print("\n1. Creating propensity score features...")

# Features that predict team change (for causal inference)
propensity_features = analysis_data[analysis_data['relative_year'] == -1].copy()

# Create pre-transition trajectory
propensity_features['performance_trajectory'] = (
    propensity_features.groupby('player_id')['primary_metric']
    .transform(lambda x: x.diff().mean())
)

# Variance in performance (inconsistency)
propensity_features['performance_variance'] = (
    propensity_features.groupby('player_id')['primary_metric']
    .transform('std')
)

# Contract year indicator (approximation based on age patterns)
propensity_features['likely_contract_year'] = (
    ((propensity_features['position'] == 'RB') & (propensity_features['age'].isin([26, 27, 28]))) |
    ((propensity_features['position'] == 'QB') & (propensity_features['age'].isin([29, 30, 31])))
).astype(int)

propensity_features.to_csv(ML_DIR / "propensity_features.csv", index=False)
print(f"   Created propensity features for {len(propensity_features)} player-seasons")

################################################################################
# 2. CAREER TRAJECTORY FEATURES
################################################################################

print("\n2. Engineering career trajectory features...")

def create_trajectory_features(df):
    """Create polynomial and exponential career trajectory features."""
    
    # Age polynomial features (capture non-linear aging)
    poly = PolynomialFeatures(degree=3, include_bias=False)
    age_poly = poly.fit_transform(df[['age']].values)
    
    df['age_squared'] = age_poly[:, 1]
    df['age_cubed'] = age_poly[:, 2]
    
    # Experience features
    df['experience_squared'] = df['experience'] ** 2
    
    # Peak age indicators by position
    df['years_from_peak'] = 0
    df.loc[df['position'] == 'RB', 'years_from_peak'] = df.loc[df['position'] == 'RB', 'age'] - 24
    df.loc[df['position'] == 'QB', 'years_from_peak'] = df.loc[df['position'] == 'QB', 'age'] - 28
    df.loc[df['position'].isin(['WR', 'TE']), 'years_from_peak'] = df.loc[df['position'].isin(['WR', 'TE']), 'age'] - 27
    
    # Exponential decay features
    df['age_exp_decay'] = np.exp(-0.1 * df['age'])
    df['years_from_peak_squared'] = df['years_from_peak'] ** 2
    
    return df

trajectory_data = create_trajectory_features(analysis_data.copy())
trajectory_data[['player_id', 'season', 'age', 'age_squared', 'age_cubed', 
                 'years_from_peak', 'age_exp_decay']].to_csv(
    ML_DIR / "trajectory_features.csv", index=False
)

print("   Created polynomial and exponential age features")

################################################################################
# 3. ROLLING WINDOW STATISTICS
################################################################################

print("\n3. Creating rolling window features...")

def create_rolling_features(df, window_sizes=[2, 3, 4]):
    """Create rolling statistics for capturing trends."""
    
    df = df.sort_values(['player_id', 'season'])
    
    for window in window_sizes:
        # Rolling mean
        df[f'primary_metric_roll{window}'] = (
            df.groupby('player_id')['primary_metric']
            .transform(lambda x: x.rolling(window, min_periods=1).mean())
        )
        
        # Rolling std (volatility)
        df[f'primary_metric_vol{window}'] = (
            df.groupby('player_id')['primary_metric']
            .transform(lambda x: x.rolling(window, min_periods=1).std())
        )
        
        # Rolling min/max (range)
        df[f'primary_metric_range{window}'] = (
            df.groupby('player_id')['primary_metric']
            .transform(lambda x: x.rolling(window, min_periods=1).max() - 
                                 x.rolling(window, min_periods=1).min())
        )
    
    return df

rolling_data = create_rolling_features(analysis_data.copy())
rolling_cols = [col for col in rolling_data.columns if 'roll' in col or 'vol' in col or 'range' in col]
rolling_data[['player_id', 'season'] + rolling_cols].to_csv(
    ML_DIR / "rolling_features.csv", index=False
)

print(f"   Created {len(rolling_cols)} rolling window features")

################################################################################
# 4. MOMENTUM AND TREND FEATURES
################################################################################

print("\n4. Creating momentum indicators...")

def create_momentum_features(df):
    """Create features capturing performance trends and momentum."""
    
    df = df.sort_values(['player_id', 'season'])
    
    # First-order momentum (velocity)
    df['performance_momentum'] = (
        df.groupby('player_id')['primary_metric'].transform(lambda x: x.diff())
    )
    
    # Second-order momentum (acceleration)
    df['performance_acceleration'] = (
        df.groupby('player_id')['performance_momentum'].transform(lambda x: x.diff())
    )
    
    # Cumulative performance change from baseline
    df['cumulative_change'] = (
        df.groupby('player_id')['primary_metric']
        .transform(lambda x: x - x.iloc[0] if len(x) > 0 else 0)
    )
    
    # Consecutive improvement/decline streaks
    df['improvement'] = (df['performance_momentum'] > 0).astype(int)
    df['improvement_streak'] = (
        df.groupby('player_id')['improvement']
        .transform(lambda x: x.groupby((x != x.shift()).cumsum()).cumsum())
    )
    
    return df

momentum_data = create_momentum_features(analysis_data.copy())
momentum_cols = ['performance_momentum', 'performance_acceleration', 
                 'cumulative_change', 'improvement_streak']
momentum_data[['player_id', 'season'] + momentum_cols].to_csv(
    ML_DIR / "momentum_features.csv", index=False
)

print("   Created momentum and acceleration features")

################################################################################
# 5. INTERACTION FEATURES
################################################################################

print("\n5. Creating interaction features...")

def create_interaction_features(df):
    """Create meaningful interaction terms."""
    
    # Age × Post interaction (test if older players benefit more/less)
    df['age_post_interaction'] = df['age'] * df['post']
    
    # Position × Post interactions
    for pos in ['QB', 'RB', 'WR', 'TE']:
        df[f'{pos}_post'] = (df['position'] == pos).astype(int) * df['post']
    
    # Performance trend × Post (do declining players rebound?)
    if 'performance_momentum' in df.columns:
        df['momentum_post_interaction'] = df['performance_momentum'] * df['post']
    
    # Team quality × Post (does team quality moderate effect?)
    if 'team_ypc_excl' in df.columns:
        df['team_quality_post'] = df['team_ypc_excl'].fillna(0) * df['post']
    
    return df

interaction_data = create_interaction_features(momentum_data.copy())
interaction_cols = [col for col in interaction_data.columns if 'interaction' in col or '_post' in col]
interaction_data[['player_id', 'season'] + interaction_cols].to_csv(
    ML_DIR / "interaction_features.csv", index=False
)

print(f"   Created {len(interaction_cols)} interaction terms")

################################################################################
# 6. DIMENSIONALITY REDUCTION (PCA)
################################################################################

print("\n6. Applying PCA for dimensionality reduction...")

# Select numeric features for PCA
numeric_cols = analysis_data.select_dtypes(include=[np.number]).columns
feature_cols = [col for col in numeric_cols if col not in [
    'player_id', 'season', 'transition_year', 'relative_year', 'post'
]]

# Handle missing values
pca_data = analysis_data[feature_cols].fillna(analysis_data[feature_cols].median())

# Standardize
scaler = StandardScaler()
pca_data_scaled = scaler.fit_transform(pca_data)

# Apply PCA
pca = PCA(n_components=min(10, len(feature_cols)))
pca_components = pca.fit_transform(pca_data_scaled)

# Create DataFrame
pca_df = pd.DataFrame(
    pca_components,
    columns=[f'PC{i+1}' for i in range(pca_components.shape[1])]
)
pca_df['player_id'] = analysis_data['player_id'].values
pca_df['season'] = analysis_data['season'].values

pca_df.to_csv(ML_DIR / "pca_features.csv", index=False)

# Save explained variance
explained_var = pd.DataFrame({
    'Component': [f'PC{i+1}' for i in range(len(pca.explained_variance_ratio_))],
    'Explained_Variance': pca.explained_variance_ratio_,
    'Cumulative_Variance': np.cumsum(pca.explained_variance_ratio_)
})
explained_var.to_csv(ML_DIR / "pca_explained_variance.csv", index=False)

print(f"   Created {pca_components.shape[1]} principal components")
print(f"   Cumulative variance explained: {explained_var['Cumulative_Variance'].iloc[-1]:.2%}")

################################################################################
# 7. TARGET ENCODING FOR CATEGORICAL VARIABLES
################################################################################

print("\n7. Creating target-encoded features...")

def target_encode(df, cat_col, target_col, smoothing=1.0):
    """Target encoding with smoothing to prevent overfitting."""
    
    # Global mean
    global_mean = df[target_col].mean()
    
    # Group statistics
    agg = df.groupby(cat_col)[target_col].agg(['count', 'mean'])
    
    # Smoothing formula
    smoothed = (agg['count'] * agg['mean'] + smoothing * global_mean) / (agg['count'] + smoothing)
    
    return df[cat_col].map(smoothed).fillna(global_mean)

target_encoded_data = analysis_data.copy()

# Team encoding (captures team quality)
target_encoded_data['team_encoded'] = target_encode(
    target_encoded_data, 'recent_team', 'primary_metric_z', smoothing=5.0
)

# Position encoding
target_encoded_data['position_encoded'] = target_encode(
    target_encoded_data, 'position', 'primary_metric_z', smoothing=10.0
)

target_encoded_data[['player_id', 'season', 'team_encoded', 'position_encoded']].to_csv(
    ML_DIR / "target_encoded_features.csv", index=False
)

print("   Created target-encoded features for team and position")

################################################################################
# 8. CREATE MASTER FEATURE SET
################################################################################

print("\n8. Assembling master feature set...")

# Merge all features
master_features = analysis_data[['player_id', 'season', 'position', 
                                  'post', 'primary_metric_z']].copy()

# Add trajectory features
master_features = master_features.merge(
    trajectory_data[['player_id', 'season', 'age_squared', 'age_cubed', 'years_from_peak']],
    on=['player_id', 'season'],
    how='left'
)

# Add rolling features (select subset)
master_features = master_features.merge(
    rolling_data[['player_id', 'season', 'primary_metric_roll2', 'primary_metric_vol3']],
    on=['player_id', 'season'],
    how='left'
)

# Add momentum
master_features = master_features.merge(
    momentum_data[['player_id', 'season', 'performance_momentum', 'performance_acceleration']],
    on=['player_id', 'season'],
    how='left'
)

# Add PCA
master_features = master_features.merge(
    pca_df[['player_id', 'season', 'PC1', 'PC2', 'PC3']],
    on=['player_id', 'season'],
    how='left'
)

# Add target encoding
master_features = master_features.merge(
    target_encoded_data[['player_id', 'season', 'team_encoded', 'position_encoded']],
    on=['player_id', 'season'],
    how='left'
)

master_features.to_csv(ML_DIR / "master_features.csv", index=False)
master_features.to_parquet(ML_DIR / "master_features.parquet")

print(f"\n   Master feature set: {len(master_features)} rows × {len(master_features.columns)} columns")

################################################################################
# 9. CREATE TRAIN/TEST SPLIT (by player)
################################################################################

print("\n9. Creating train/test split...")

# Split by player to avoid leakage
unique_players = master_features['player_id'].unique()
np.random.seed(42)
test_players = np.random.choice(unique_players, size=int(len(unique_players) * 0.2), replace=False)

train_data = master_features[~master_features['player_id'].isin(test_players)]
test_data = master_features[master_features['player_id'].isin(test_players)]

train_data.to_csv(ML_DIR / "train_features.csv", index=False)
test_data.to_csv(ML_DIR / "test_features.csv", index=False)

print(f"   Train: {len(train_data)} obs ({train_data['player_id'].nunique()} players)")
print(f"   Test:  {len(test_data)} obs ({test_data['player_id'].nunique()} players)")

################################################################################
# 10. FEATURE IMPORTANCE METADATA
################################################################################

print("\n10. Generating feature metadata...")

feature_metadata = pd.DataFrame({
    'feature': master_features.columns,
    'dtype': master_features.dtypes.values,
    'missing_pct': (master_features.isnull().sum() / len(master_features) * 100).values,
    'unique_values': [master_features[col].nunique() for col in master_features.columns],
    'feature_type': ['id', 'id', 'categorical', 'binary', 'target'] + 
                    ['engineered'] * (len(master_features.columns) - 5)
})

feature_metadata.to_csv(ML_DIR / "feature_metadata.csv", index=False)

print("\n" + "=" * 80)
print("FEATURE ENGINEERING COMPLETE")
print("=" * 80)
print(f"\nAll features saved to: {ML_DIR}")
print(f"Ready for predictive modeling (modeling.py)")

if __name__ == "__main__":
    print("\nFeature engineering pipeline completed!")
    print("Next step: Run modeling.py for ML predictions")
