"""
predictive_modeling.py
NFL Veterans Team Change Analysis
Purpose: Machine learning predictions complementing R statistical inference

While R scripts focus on causal inference and hypothesis testing,
this script focuses on:
1. Predicting which veterans will succeed after team changes
2. Identifying high-impact transitions using ML
3. Feature importance and interpretability
4. Out-of-sample prediction performance
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Directories
ML_DIR = Path("data/ml_features")
MODELS_DIR = Path("outputs/models/ml")
FIGS_DIR = Path("outputs/figures")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("PREDICTIVE MODELING - ML APPROACH")
print("Complementing R's causal inference with predictive performance")
print("=" * 80)

################################################################################
# 1. LOAD DATA
################################################################################

print("\n1. Loading feature-engineered data...")

try:
    train_data = pd.read_csv(ML_DIR / "train_features.csv")
    test_data = pd.read_csv(ML_DIR / "test_features.csv")
    print(f"   Train: {len(train_data)} observations")
    print(f"   Test:  {len(test_data)} observations")
except FileNotFoundError:
    print("Error: Run feature_engineering.py first")
    exit(1)

# Separate features and target
feature_cols = [col for col in train_data.columns if col not in [
    'player_id', 'season', 'position', 'primary_metric_z'
]]

X_train = train_data[feature_cols].fillna(0)
y_train = train_data['primary_metric_z'].fillna(0)

X_test = test_data[feature_cols].fillna(0)
y_test = test_data['primary_metric_z'].fillna(0)

print(f"   Features: {len(feature_cols)}")

################################################################################
# 2. BASELINE: LINEAR MODELS
################################################################################

print("\n2. Training baseline linear models...")

# Ridge regression
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)
ridge_r2 = r2_score(y_test, ridge_pred)
ridge_rmse = np.sqrt(mean_squared_error(y_test, ridge_pred))

print(f"   Ridge - R²: {ridge_r2:.4f}, RMSE: {ridge_rmse:.4f}")

# Lasso
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
lasso_pred = lasso.predict(X_test)
lasso_r2 = r2_score(y_test, lasso_pred)
lasso_rmse = np.sqrt(mean_squared_error(y_test, lasso_pred))

print(f"   Lasso - R²: {lasso_r2:.4f}, RMSE: {lasso_rmse:.4f}")

# Elastic Net
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic.fit(X_train, y_train)
elastic_pred = elastic.predict(X_test)
elastic_r2 = r2_score(y_test, elastic_pred)
elastic_rmse = np.sqrt(mean_squared_error(y_test, elastic_pred))

print(f"   ElasticNet - R²: {elastic_r2:.4f}, RMSE: {elastic_rmse:.4f}")

################################################################################
# 3. TREE-BASED MODELS
################################################################################

print("\n3. Training tree-based ensemble models...")

# Random Forest
rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=20,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_r2 = r2_score(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))

print(f"   Random Forest - R²: {rf_r2:.4f}, RMSE: {rf_rmse:.4f}")

# Gradient Boosting
gb = GradientBoostingRegressor(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)
gb.fit(X_train, y_train)
gb_pred = gb.predict(X_test)
gb_r2 = r2_score(y_test, gb_pred)
gb_rmse = np.sqrt(mean_squared_error(y_test, gb_pred))

print(f"   Gradient Boosting - R²: {gb_r2:.4f}, RMSE: {gb_rmse:.4f}")

################################################################################
# 4. ADVANCED BOOSTING MODELS
################################################################################

print("\n4. Training advanced boosting models...")

# XGBoost
xgb_model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
xgb_r2 = r2_score(y_test, xgb_pred)
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))

print(f"   XGBoost - R²: {xgb_r2:.4f}, RMSE: {xgb_rmse:.4f}")

# LightGBM
lgb_model = lgb.LGBMRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
lgb_model.fit(X_train, y_train)
lgb_pred = lgb_model.predict(X_test)
lgb_r2 = r2_score(y_test, lgb_pred)
lgb_rmse = np.sqrt(mean_squared_error(y_test, lgb_pred))

print(f"   LightGBM - R²: {lgb_r2:.4f}, RMSE: {lgb_rmse:.4f}")

################################################################################
# 5. MODEL COMPARISON
################################################################################

print("\n5. Comparing all models...")

model_comparison = pd.DataFrame({
    'Model': ['Ridge', 'Lasso', 'ElasticNet', 'Random Forest', 
              'Gradient Boosting', 'XGBoost', 'LightGBM'],
    'R2': [ridge_r2, lasso_r2, elastic_r2, rf_r2, gb_r2, xgb_r2, lgb_r2],
    'RMSE': [ridge_rmse, lasso_rmse, elastic_rmse, rf_rmse, gb_rmse, xgb_rmse, lgb_rmse],
    'MAE': [
        mean_absolute_error(y_test, ridge_pred),
        mean_absolute_error(y_test, lasso_pred),
        mean_absolute_error(y_test, elastic_pred),
        mean_absolute_error(y_test, rf_pred),
        mean_absolute_error(y_test, gb_pred),
        mean_absolute_error(y_test, xgb_pred),
        mean_absolute_error(y_test, lgb_pred)
    ]
})

model_comparison = model_comparison.sort_values('R2', ascending=False)
print(model_comparison.to_string(index=False))

model_comparison.to_csv(MODELS_DIR / "model_comparison.csv", index=False)

# Select best model
best_model_name = model_comparison.iloc[0]['Model']
print(f"\n   Best model: {best_model_name}")

################################################################################
# 6. FEATURE IMPORTANCE (from best tree model)
################################################################################

print("\n6. Analyzing feature importance...")

# Use XGBoost for feature importance (usually performs well)
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n   Top 10 most important features:")
print(feature_importance.head(10).to_string(index=False))

feature_importance.to_csv(MODELS_DIR / "feature_importance.csv", index=False)

# Plot feature importance
plt.figure(figsize=(10, 8))
plt.barh(feature_importance.head(15)['feature'], 
         feature_importance.head(15)['importance'])
plt.xlabel('Importance')
plt.title('Top 15 Most Important Features (XGBoost)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(FIGS_DIR / "feature_importance.png", dpi=300)
plt.close()

print("   Saved feature importance plot")

################################################################################
# 7. PREDICTION ERROR ANALYSIS
################################################################################

print("\n7. Analyzing prediction errors...")

# Use best model (XGBoost)
test_results = test_data[['player_id', 'season', 'position', 'post']].copy()
test_results['actual'] = y_test.values
test_results['predicted'] = xgb_pred
test_results['error'] = test_results['actual'] - test_results['predicted']
test_results['abs_error'] = np.abs(test_results['error'])

# Error by position
error_by_position = test_results.groupby('position').agg({
    'error': 'mean',
    'abs_error': 'mean',
    'player_id': 'count'
}).rename(columns={'player_id': 'n_obs'})

print("\n   Prediction error by position:")
print(error_by_position)

error_by_position.to_csv(MODELS_DIR / "error_by_position.csv")

# Error by pre/post
error_by_period = test_results.groupby('post').agg({
    'error': 'mean',
    'abs_error': 'mean',
    'player_id': 'count'
}).rename(columns={'player_id': 'n_obs', 'post': 'period'})

error_by_period.index = ['Pre-transition', 'Post-transition']
print("\n   Prediction error by period:")
print(error_by_period)

################################################################################
# 8. IDENTIFY HIGH-IMPACT PREDICTIONS
################################################################################

print("\n8. Identifying high-impact transition predictions...")

# Focus on post-transition predictions
post_predictions = test_results[test_results['post'] == 1].copy()

# Calculate surprise index (actual - predicted)
post_predictions['surprise_index'] = post_predictions['actual'] - post_predictions['predicted']

# Top positive surprises (exceeded expectations)
top_positive = post_predictions.nlargest(10, 'surprise_index')[
    ['player_id', 'position', 'predicted', 'actual', 'surprise_index']
]

print("\n   Top 10 players who exceeded predictions:")
print(top_positive.to_string(index=False))

# Top negative surprises (underperformed)
top_negative = post_predictions.nsmallest(10, 'surprise_index')[
    ['player_id', 'position', 'predicted', 'actual', 'surprise_index']
]

print("\n   Top 10 players who underperformed predictions:")
print(top_negative.to_string(index=False))

# Save
post_predictions.to_csv(MODELS_DIR / "post_transition_predictions.csv", index=False)

################################################################################
# 9. VISUALIZATIONS
################################################################################

print("\n9. Creating visualizations...")

# Actual vs Predicted scatter
plt.figure(figsize=(10, 8))
plt.scatter(test_results['actual'], test_results['predicted'], alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'r--', lw=2, label='Perfect prediction')
plt.xlabel('Actual Performance (z-score)')
plt.ylabel('Predicted Performance (z-score)')
plt.title(f'Actual vs Predicted Performance ({best_model_name})')
plt.legend()
plt.tight_layout()
plt.savefig(FIGS_DIR / "actual_vs_predicted.png", dpi=300)
plt.close()

# Residual plot
plt.figure(figsize=(10, 6))
plt.scatter(test_results['predicted'], test_results['error'], alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Performance')
plt.ylabel('Prediction Error')
plt.title('Residual Plot')
plt.tight_layout()
plt.savefig(FIGS_DIR / "residual_plot.png", dpi=300)
plt.close()

# Error distribution by position
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for idx, pos in enumerate(['QB', 'RB', 'WR', 'TE']):
    ax = axes[idx // 2, idx % 2]
    pos_errors = test_results[test_results['position'] == pos]['error']
    ax.hist(pos_errors, bins=20, edgecolor='black')
    ax.axvline(x=0, color='r', linestyle='--')
    ax.set_title(f'{pos} - MAE: {np.abs(pos_errors).mean():.3f}')
    ax.set_xlabel('Prediction Error')
    ax.set_ylabel('Frequency')
plt.tight_layout()
plt.savefig(FIGS_DIR / "error_distribution_by_position.png", dpi=300)
plt.close()

print("   Saved visualization plots")

################################################################################
# 10. SAVE BEST MODEL
################################################################################

print("\n10. Saving trained models...")

import pickle

# Save XGBoost (best model)
with open(MODELS_DIR / "xgboost_model.pkl", 'wb') as f:
    pickle.dump(xgb_model, f)

# Save feature columns for future predictions
with open(MODELS_DIR / "feature_columns.pkl", 'wb') as f:
    pickle.dump(feature_cols, f)

print("   Saved XGBoost model and feature columns")

################################################################################
# SUMMARY
################################################################################

print("\n" + "=" * 80)
print("PREDICTIVE MODELING COMPLETE")
print("=" * 80)

print(f"\nBest Model: {best_model_name}")
print(f"Test R²: {model_comparison.iloc[0]['R2']:.4f}")
print(f"Test RMSE: {model_comparison.iloc[0]['RMSE']:.4f}")

print(f"\nOutputs saved to:")
print(f"  Models: {MODELS_DIR}")
print(f"  Figures: {FIGS_DIR}")

print("\n" + "=" * 80)
print("COMPARISON: ML vs R Statistical Approach")
print("=" * 80)
print("R (lme4/nlme): Causal inference, hypothesis testing, p-values")
print("Python (XGBoost): Prediction, feature importance, out-of-sample performance")
print("\nBoth approaches complement each other for comprehensive analysis!")
print("=" * 80)

if __name__ == "__main__":
    print("\nPredictive modeling pipeline completed!")

