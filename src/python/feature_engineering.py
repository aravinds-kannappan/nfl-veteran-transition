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
import warnings
warnings.filterwarnings('ignore')

# Directories
PROCESSED_DIR = Path("data/processed")
ML_DIR = Path("data/ml_features")
ML_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("MACHINE LEARNING FEATURE ENGINEERING")
print("Complementing R statistical inference with ML prediction")
print("=" * 80)

# Load data from preprocessing
try:
    analysis_data = pd.read_csv(PROCESSED_DIR / "nfl_panel_processed.csv")
    print(f"\nLoaded {len(analysis_data):,} observations from preprocessing")
except FileNotFoundError:
    print("Error: Run preprocessing.py first")
    exit(1)

################################################################################
# 1. PROPENSITY SCORE FEATURES (for matching/weighting)
################################################################################

print("\n1. Creating propensity score features...")

propensity_features = analysis_data.copy()

# Features that predict team change
if 'gsis_id' in propensity_features.columns and 'season' in propensity_features.columns:
    # Create pre-transition trajectory from pre_trend_slope
    if 'pre_trend_slope' in propensity_features.columns:
        propensity_features['performance_trajectory'] = propensity_features['pre_trend_slope']
    
    # Performance variance
    if 'z_score' in propensity_features.columns:
        propensity_features['performance_variance'] = (
            propensity_features.groupby('gsis_id')['z_score']
            .transform('std')
        )
    
    # Contract year approximation (experience-based)
    if 'years_exp' in propensity_features.columns:
        propensity_features['likely_contract_year'] = (
            ((propensity_features['position_group'] == 'RB') & 
             (propensity_features['years_exp'].isin([6, 7, 8]))) |
            ((propensity_features['position_group'] == 'QB') & 
             (propensity_features['years_exp'].isin([7, 8, 9])))
        ).astype(int)
    
    propensity_features.to_csv(ML_DIR / "propensity_features.csv", index=False)
    print(f"   Created propensity features for {len(propensity_features):,} records")

################################################################################
# 2. CAREER TRAJECTORY FEATURES
################################################################################

print("\n2. Engineering career trajectory features...")

trajectory_data = analysis_data.copy()

if 'age' in trajectory_data.columns:
    # Age polynomial features (capture non-linear aging)
    poly = PolynomialFeatures(degree=3, include_bias=False)
    
    # Handle missing values
    age_data = trajectory_data[['age']].fillna(trajectory_data['age'].median())
    age_poly = poly.fit_transform(age_data)
    
    trajectory_data['age_squared'] = age_poly[:, 1]
    trajectory_data['age_cubed'] = age_poly[:, 2]
    
    # Exponential decay features
    trajectory_data['age_exp_decay'] = np.exp(-0.1 * trajectory_data['age'])
    
    print("   Created polynomial and exponential age features")

# Experience-based trajectory
if 'years_exp' in trajectory_data.columns:
    trajectory_data['exp_squared'] = trajectory_data['years_exp'] ** 2
    trajectory_data['peak_years_away'] = trajectory_data['years_exp'] - 5
    print("   Created experience-based trajectory features")

trajectory_data.to_csv(ML_DIR / "trajectory_features.csv", index=False)

################################################################################
# 3. ROLLING WINDOW STATISTICS
################################################################################

print("\n3. Creating rolling window features...")

rolling_data = analysis_data.copy()

if 'gsis_id' in rolling_data.columns and 'season' in rolling_data.columns:
    rolling_data = rolling_data.sort_values(['gsis_id', 'season'])
    
    # Create rolling stats for key metrics
    window_sizes = [2, 3]
    
    for window in window_sizes:
        # Rolling mean for z_score
        if 'z_score' in rolling_data.columns:
            rolling_data[f'z_score_roll{window}'] = (
                rolling_data.groupby('gsis_id')['z_score']
                .transform(lambda x: x.rolling(window, min_periods=1).mean())
            )
            
            # Rolling volatility
            rolling_data[f'z_score_vol{window}'] = (
                rolling_data.groupby('gsis_id')['z_score']
                .transform(lambda x: x.rolling(window, min_periods=1).std())
            )
    
    print(f"   Created rolling window features")

rolling_data.to_csv(ML_DIR / "rolling_features.csv", index=False)

################################################################################
# 4. MOMENTUM AND TREND FEATURES
################################################################################

print("\n4. Creating momentum indicators...")

momentum_data = analysis_data.copy()

if 'gsis_id' in momentum_data.columns and 'season' in momentum_data.columns:
    momentum_data = momentum_data.sort_values(['gsis_id', 'season'])
    
    # Performance momentum (year-over-year change)
    if 'z_score' in momentum_data.columns:
        momentum_data['z_score_momentum'] = (
            momentum_data.groupby('gsis_id')['z_score'].transform(lambda x: x.diff())
        )
        
        # Acceleration (second derivative)
        momentum_data['z_score_acceleration'] = (
            momentum_data.groupby('gsis_id')['z_score_momentum'].transform(lambda x: x.diff())
        )
        
        # Cumulative change
        momentum_data['z_score_cumulative_change'] = (
            momentum_data.groupby('gsis_id')['z_score']
            .transform(lambda x: x - x.iloc[0] if len(x) > 0 else 0)
        )
    
    print("   Created momentum and acceleration features")

momentum_data.to_csv(ML_DIR / "momentum_features.csv", index=False)

################################################################################
# 5. INTERACTION FEATURES
################################################################################

print("\n5. Creating interaction features...")

interaction_data = analysis_data.copy()

interaction_cols = []

# Age × Post-transition interaction
if 'age' in interaction_data.columns and 'post_transition' in interaction_data.columns:
    interaction_data['age_post_interaction'] = (
        interaction_data['age'] * interaction_data['post_transition']
    )
    interaction_cols.append('age_post_interaction')

# Experience × Post-transition interaction
if 'years_exp' in interaction_data.columns and 'post_transition' in interaction_data.columns:
    interaction_data['exp_post_interaction'] = (
        interaction_data['years_exp'] * interaction_data['post_transition']
    )
    interaction_cols.append('exp_post_interaction')

# Team quality × Performance interaction
if 'team_quality' in interaction_data.columns and 'z_score' in interaction_data.columns:
    interaction_data['quality_performance_interaction'] = (
        interaction_data['team_quality'] * interaction_data['z_score']
    )
    interaction_cols.append('quality_performance_interaction')

# Position-specific post-transition effects
if 'position_group' in interaction_data.columns and 'post_transition' in interaction_data.columns:
    positions = interaction_data['position_group'].unique()[:4]
    for pos in positions:
        col_name = f'{pos}_post'
        interaction_data[col_name] = (
            (interaction_data['position_group'] == pos).astype(int) * interaction_data['post_transition']
        )
        interaction_cols.append(col_name)

if len(interaction_cols) > 0:
    interaction_data.to_csv(ML_DIR / "interaction_features.csv", index=False)
    print(f"   Created {len(interaction_cols)} interaction terms")

################################################################################
# 6. DIMENSIONALITY REDUCTION (PCA)
################################################################################

print("\n6. Applying PCA for dimensionality reduction...")

# Select numeric features for PCA
exclude_cols = {'gsis_id', 'season', 'position_group', 'team', 'prev_team', 'new_team',
                'changed_team', 'post_transition', 'phase', 'rel_time', 'years_since_change',
                'transition_season', 'performance_level', 'career_phase', 'exp_group',
                'adjustment_phase', 'pre_trend_category', 'data_split'}

numeric_cols = analysis_data.select_dtypes(include=[np.number]).columns
feature_cols = [col for col in numeric_cols if col not in exclude_cols]

if len(feature_cols) > 2:
    # Handle missing values
    pca_data = analysis_data[feature_cols].fillna(analysis_data[feature_cols].median())
    
    # Standardize
    scaler = StandardScaler()
    pca_data_scaled = scaler.fit_transform(pca_data)
    
    # Apply PCA
    n_components = min(10, len(feature_cols))
    pca = PCA(n_components=n_components)
    pca_components = pca.fit_transform(pca_data_scaled)
    
    # Create DataFrame
    pca_df = pd.DataFrame(
        pca_components,
        columns=[f'PC{i+1}' for i in range(pca_components.shape[1])]
    )
    
    # Add identifiers
    pca_df['gsis_id'] = analysis_data['gsis_id'].values
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

target_encoded_data = analysis_data.copy()

# Identify numeric target
if 'z_score' in target_encoded_data.columns:
    target_col = 'z_score'
    
    def target_encode(df, cat_col, target_col, smoothing=1.0):
        """Target encoding with smoothing to prevent overfitting."""
        
        if df[cat_col].dtype != 'object':
            return df[cat_col]
        
        # Global mean
        global_mean = df[target_col].mean()
        
        # Group statistics
        agg = df.groupby(cat_col)[target_col].agg(['count', 'mean'])
        
        # Smoothing formula
        smoothed = (agg['count'] * agg['mean'] + smoothing * global_mean) / (agg['count'] + smoothing)
        
        return df[cat_col].map(smoothed).fillna(global_mean)
    
    # Team encoding
    if 'team' in target_encoded_data.columns:
        target_encoded_data['team_encoded'] = target_encode(
            target_encoded_data, 'team', target_col, smoothing=5.0
        )
    
    # Position encoding
    if 'position_group' in target_encoded_data.columns:
        target_encoded_data['position_encoded'] = target_encode(
            target_encoded_data, 'position_group', target_col, smoothing=10.0
        )
    
    target_encoded_data.to_csv(ML_DIR / "target_encoded_features.csv", index=False)
    print("   Created target-encoded features")

################################################################################
# 8. CREATE MASTER FEATURE SET
################################################################################

print("\n8. Assembling master feature set...")

master_features = analysis_data.copy()

print(f"   Master feature set: {len(master_features):,} rows × {len(master_features.columns)} columns")

master_features.to_csv(ML_DIR / "master_features.csv", index=False)
master_features.to_parquet(ML_DIR / "master_features.parquet")

print(f"   ✓ Saved master features")

################################################################################
# 9. CREATE TRAIN/TEST SPLIT
################################################################################

print("\n9. Creating train/test split...")

if 'data_split' in master_features.columns:
    train_data = master_features[master_features['data_split'] == 'train']
    test_data = master_features[master_features['data_split'].isin(['validation', 'test'])]
else:
    # Fallback: random split by player
    if 'gsis_id' in master_features.columns:
        unique_players = master_features['gsis_id'].unique()
        np.random.seed(42)
        test_players = np.random.choice(unique_players, size=int(len(unique_players) * 0.2), replace=False)
        
        train_data = master_features[~master_features['gsis_id'].isin(test_players)]
        test_data = master_features[master_features['gsis_id'].isin(test_players)]
    else:
        train_data = master_features
        test_data = master_features

train_data.to_csv(ML_DIR / "train_features.csv", index=False)
test_data.to_csv(ML_DIR / "test_features.csv", index=False)

print(f"   Train: {len(train_data):,} obs")
print(f"   Test:  {len(test_data):,} obs")

################################################################################
# 10. FEATURE METADATA
################################################################################

print("\n10. Generating feature metadata...")

feature_metadata = pd.DataFrame({
    'feature': master_features.columns,
    'dtype': master_features.dtypes.values,
    'missing_pct': (master_features.isnull().sum() / len(master_features) * 100).values,
    'unique_values': [master_features[col].nunique() for col in master_features.columns],
})

feature_metadata.to_csv(ML_DIR / "feature_metadata.csv", index=False)

print("\n" + "=" * 80)
print("FEATURE ENGINEERING COMPLETE")
print("=" * 80)
print(f"\nAll features saved to: {ML_DIR}")
print(f"\nFiles created:")
print(f"  ✓ master_features.csv/parquet")
print(f"  ✓ train_features.csv")
print(f"  ✓ test_features.csv")
print(f"  ✓ propensity_features.csv")
print(f"  ✓ trajectory_features.csv")
print(f"  ✓ rolling_features.csv")
print(f"  ✓ momentum_features.csv")
print(f"  ✓ interaction_features.csv")
print(f"  ✓ pca_features.csv")
print(f"  ✓ target_encoded_features.csv")
print(f"  ✓ feature_metadata.csv")
print(f"\nReady for predictive modeling (modeling.py)")
print("=" * 80 + "\n")

if __name__ == "__main__":
    print("Feature engineering pipeline completed!")
    print("Next step: Run modeling.py for ML predictions")
