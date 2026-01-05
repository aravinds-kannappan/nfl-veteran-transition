"""
modeling.py
NFL Veterans Team Change Analysis
Purpose: Machine learning predictions complementing R statistical inference

This script focuses on:
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
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Directories
ML_DIR = Path("data/ml_features")
MODELS_DIR = Path("outputs/models")
ANALYSIS_DIR = Path("outputs/analysis")
FIGS_DIR = Path("outputs/figures")

# Create all directories
MODELS_DIR.mkdir(parents=True, exist_ok=True)
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
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
    print(f"   ✓ Train: {len(train_data):,} observations")
    print(f"   ✓ Test:  {len(test_data):,} observations")
except FileNotFoundError:
    print("   ❌ Error: Run feature_engineering.py first")
    exit(1)

################################################################################
# 2. PREPARE FEATURES AND TARGET
################################################################################

print("\n2. Preparing features and target...")

# Identify numeric columns for features
exclude_cols = {'gsis_id', 'season', 'position_group', 'team', 'prev_team', 'new_team',
                'changed_team', 'post_transition', 'phase', 'rel_time', 'years_since_change',
                'transition_season', 'performance_level', 'career_phase', 'exp_group',
                'adjustment_phase', 'pre_trend_category', 'data_split', 'volume'}

# Find target column (z_score is our performance target)
target_col = 'z_score'

numeric_cols = train_data.select_dtypes(include=[np.number]).columns
feature_cols = [col for col in numeric_cols if col not in exclude_cols and col != target_col]

print(f"   Target variable: {target_col}")
print(f"   Features: {len(feature_cols)}")

# Prepare training data
X_train = train_data[feature_cols].fillna(0)
y_train = train_data[target_col].fillna(0)

X_test = test_data[feature_cols].fillna(0)
y_test = test_data[target_col].fillna(0)

# Remove rows with NaN target
mask_train = y_train.notna()
mask_test = y_test.notna()

X_train = X_train[mask_train]
y_train = y_train[mask_train]
X_test = X_test[mask_test]
y_test = y_test[mask_test]

print(f"   After cleaning - Train: {len(X_train):,}, Test: {len(X_test):,}")

################################################################################
# 3. BASELINE: LINEAR MODELS
################################################################################

print("\n3. Training baseline linear models...")

results_list = []

# Ridge regression
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)
ridge_r2 = r2_score(y_test, ridge_pred)
ridge_rmse = np.sqrt(mean_squared_error(y_test, ridge_pred))
ridge_mae = mean_absolute_error(y_test, ridge_pred)

print(f"   Ridge - R²: {ridge_r2:.4f}, RMSE: {ridge_rmse:.4f}, MAE: {ridge_mae:.4f}")
results_list.append(['Ridge', ridge_r2, ridge_rmse, ridge_mae, ridge_pred])

# Lasso
lasso = Lasso(alpha=0.1, max_iter=5000)
lasso.fit(X_train, y_train)
lasso_pred = lasso.predict(X_test)
lasso_r2 = r2_score(y_test, lasso_pred)
lasso_rmse = np.sqrt(mean_squared_error(y_test, lasso_pred))
lasso_mae = mean_absolute_error(y_test, lasso_pred)

print(f"   Lasso - R²: {lasso_r2:.4f}, RMSE: {lasso_rmse:.4f}, MAE: {lasso_mae:.4f}")
results_list.append(['Lasso', lasso_r2, lasso_rmse, lasso_mae, lasso_pred])

# Elastic Net
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000)
elastic.fit(X_train, y_train)
elastic_pred = elastic.predict(X_test)
elastic_r2 = r2_score(y_test, elastic_pred)
elastic_rmse = np.sqrt(mean_squared_error(y_test, elastic_pred))
elastic_mae = mean_absolute_error(y_test, elastic_pred)

print(f"   ElasticNet - R²: {elastic_r2:.4f}, RMSE: {elastic_rmse:.4f}, MAE: {elastic_mae:.4f}")
results_list.append(['ElasticNet', elastic_r2, elastic_rmse, elastic_mae, elastic_pred])

################################################################################
# 4. TREE-BASED MODELS
################################################################################

print("\n4. Training tree-based ensemble models...")

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
rf_mae = mean_absolute_error(y_test, rf_pred)

print(f"   Random Forest - R²: {rf_r2:.4f}, RMSE: {rf_rmse:.4f}, MAE: {rf_mae:.4f}")
results_list.append(['Random Forest', rf_r2, rf_rmse, rf_mae, rf_pred])

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
gb_mae = mean_absolute_error(y_test, gb_pred)

print(f"   Gradient Boosting - R²: {gb_r2:.4f}, RMSE: {gb_rmse:.4f}, MAE: {gb_mae:.4f}")
results_list.append(['Gradient Boosting', gb_r2, gb_rmse, gb_mae, gb_pred])

################################################################################
# 5. ADVANCED BOOSTING MODELS
################################################################################

print("\n5. Training advanced boosting models...")

xgb_available = False
lgb_available = False

# XGBoost
try:
    xgb_model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    xgb_r2 = r2_score(y_test, xgb_pred)
    xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
    xgb_mae = mean_absolute_error(y_test, xgb_pred)
    
    print(f"   XGBoost - R²: {xgb_r2:.4f}, RMSE: {xgb_rmse:.4f}, MAE: {xgb_mae:.4f}")
    results_list.append(['XGBoost', xgb_r2, xgb_rmse, xgb_mae, xgb_pred])
    xgb_available = True
except Exception as e:
    print(f"   ⚠ XGBoost failed: {str(e)}")

# LightGBM
try:
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
    lgb_mae = mean_absolute_error(y_test, lgb_pred)
    
    print(f"   LightGBM - R²: {lgb_r2:.4f}, RMSE: {lgb_rmse:.4f}, MAE: {lgb_mae:.4f}")
    results_list.append(['LightGBM', lgb_r2, lgb_rmse, lgb_mae, lgb_pred])
    lgb_available = True
except Exception as e:
    print(f"   ⚠ LightGBM failed: {str(e)}")

################################################################################
# 6. MODEL COMPARISON
################################################################################

print("\n6. Comparing all models...")

model_comparison = pd.DataFrame(
    [[r[0], r[1], r[2], r[3]] for r in results_list],
    columns=['Model', 'R2', 'RMSE', 'MAE']
)
model_comparison = model_comparison.sort_values('R2', ascending=False)

print("\n" + model_comparison.to_string(index=False))

model_comparison.to_csv(ANALYSIS_DIR / "model_comparison.csv", index=False)

# Select best model
best_model_name = model_comparison.iloc[0]['Model']
best_r2 = model_comparison.iloc[0]['R2']
best_predictions = [r[4] for r in results_list if r[0] == best_model_name][0]

print(f"\n   ✓ Best model: {best_model_name} (R² = {best_r2:.4f})")

################################################################################
# 7. FEATURE IMPORTANCE
################################################################################

print("\n7. Analyzing feature importance...")

# Get feature importance from best tree-based model
if best_model_name == 'XGBoost' and xgb_available:
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
elif best_model_name == 'LightGBM' and lgb_available:
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': lgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
elif best_model_name == 'Random Forest':
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
elif best_model_name == 'Gradient Boosting':
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': gb.feature_importances_
    }).sort_values('importance', ascending=False)
else:
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': [1.0] * len(feature_cols)
    })

print("\n   Top 10 most important features:")
print(feature_importance.head(10).to_string(index=False))

feature_importance.to_csv(ANALYSIS_DIR / "feature_importance.csv", index=False)

# Plot feature importance
if len(feature_importance) > 0:
    plt.figure(figsize=(10, 8))
    top_n = min(15, len(feature_importance))
    plt.barh(feature_importance.head(top_n)['feature'], 
             feature_importance.head(top_n)['importance'],
             color='steelblue')
    plt.xlabel('Importance Score')
    plt.title(f'Top {top_n} Most Important Features ({best_model_name})')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(FIGS_DIR / "feature_importance.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: feature_importance.png")

################################################################################
# 8. PREDICTION ANALYSIS
################################################################################

print("\n8. Analyzing predictions...")

# Create results dataframe
test_results = pd.DataFrame({
    'actual': y_test.values,
    'predicted': best_predictions
})

test_results['error'] = test_results['actual'] - test_results['predicted']
test_results['abs_error'] = np.abs(test_results['error'])

# Add position info if available
if 'position_group' in test_data.columns:
    test_results['position'] = test_data.loc[test_data.index.isin(X_test.index), 'position_group'].values

print(f"\n   Prediction statistics:")
print(f"     Mean Error: {test_results['error'].mean():.4f}")
print(f"     Mean Abs Error: {test_results['abs_error'].mean():.4f}")
print(f"     Std Error: {test_results['error'].std():.4f}")

test_results.to_csv(ANALYSIS_DIR / "predictions.csv", index=False)
print(f"   ✓ Saved: predictions.csv")

################################################################################
# 9. VISUALIZATIONS
################################################################################

print("\n9. Creating visualizations...")

# Actual vs Predicted scatter
plt.figure(figsize=(10, 8))
plt.scatter(test_results['actual'], test_results['predicted'], alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
min_val = min(test_results['actual'].min(), test_results['predicted'].min())
max_val = max(test_results['actual'].max(), test_results['predicted'].max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
plt.xlabel(f'Actual {target_col}', fontsize=12)
plt.ylabel(f'Predicted {target_col}', fontsize=12)
plt.title(f'Actual vs Predicted Performance ({best_model_name})', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIGS_DIR / "actual_vs_predicted.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: actual_vs_predicted.png")

# Residual plot
plt.figure(figsize=(10, 6))
plt.scatter(test_results['predicted'], test_results['error'], alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel('Predicted Value', fontsize=12)
plt.ylabel('Prediction Error', fontsize=12)
plt.title('Residual Plot', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIGS_DIR / "residual_plot.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: residual_plot.png")

# Model comparison
plt.figure(figsize=(12, 6))
x_pos = np.arange(len(model_comparison))
plt.bar(x_pos, model_comparison['R2'], color='steelblue', alpha=0.8, edgecolor='black')
plt.xlabel('Model', fontsize=12)
plt.ylabel('R² Score', fontsize=12)
plt.title('Model Performance Comparison (R² on Test Set)', fontsize=14, fontweight='bold')
plt.xticks(x_pos, model_comparison['Model'], rotation=45, ha='right')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(FIGS_DIR / "model_comparison.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: model_comparison.png")

# Error distribution
plt.figure(figsize=(10, 6))
plt.hist(test_results['error'], bins=30, edgecolor='black', alpha=0.7, color='steelblue')
plt.axvline(x=0, color='r', linestyle='--', lw=2, label='Zero Error')
plt.xlabel('Prediction Error', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Distribution of Prediction Errors', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(FIGS_DIR / "error_distribution.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: error_distribution.png")

################################################################################
# 10. SAVE MODEL
################################################################################

print("\n10. Saving trained model...")

import pickle

# Save best model
if best_model_name == 'XGBoost' and xgb_available:
    with open(MODELS_DIR / "best_model_xgboost.pkl", 'wb') as f:
        pickle.dump(xgb_model, f)
elif best_model_name == 'LightGBM' and lgb_available:
    with open(MODELS_DIR / "best_model_lightgbm.pkl", 'wb') as f:
        pickle.dump(lgb_model, f)
elif best_model_name == 'Random Forest':
    with open(MODELS_DIR / "best_model_rf.pkl", 'wb') as f:
        pickle.dump(rf, f)
elif best_model_name == 'Gradient Boosting':
    with open(MODELS_DIR / "best_model_gb.pkl", 'wb') as f:
        pickle.dump(gb, f)

# Save feature columns
with open(MODELS_DIR / "feature_columns.pkl", 'wb') as f:
    pickle.dump(feature_cols, f)

print(f"   ✓ Saved: best_model and feature_columns.pkl")

################################################################################
# SUMMARY
################################################################################

print("\n" + "=" * 80)
print("PREDICTIVE MODELING COMPLETE")
print("=" * 80)

print(f"\nBest Model: {best_model_name}")
print(f"Test R²: {model_comparison.iloc[0]['R2']:.4f}")
print(f"Test RMSE: {model_comparison.iloc[0]['RMSE']:.4f}")
print(f"Test MAE: {model_comparison.iloc[0]['MAE']:.4f}")

print(f"\nOutput directories created:")
print(f"  ✓ {MODELS_DIR}")
print(f"  ✓ {ANALYSIS_DIR}")
print(f"  ✓ {FIGS_DIR}")

print(f"\nFiles saved:")
print(f"  Models: {MODELS_DIR}/")
print(f"    • best_model_*.pkl")
print(f"    • feature_columns.pkl")
print(f"  Analysis: {ANALYSIS_DIR}/")
print(f"    • model_comparison.csv")
print(f"    • feature_importance.csv")
print(f"    • predictions.csv")
print(f"  Figures: {FIGS_DIR}/")
print(f"    • feature_importance.png")
print(f"    • actual_vs_predicted.png")
print(f"    • residual_plot.png")
print(f"    • model_comparison.png")
print(f"    • error_distribution.png")

print("\n" + "=" * 80)
print("COMPARISON: ML vs R Statistical Approach")
print("=" * 80)
print("R (lme4/nlme): Causal inference, hypothesis testing, p-values")
print("Python (ML): Prediction, feature importance, out-of-sample performance")
print("\nBoth approaches complement each other for comprehensive analysis!")
print("=" * 80 + "\n")

if __name__ == "__main__":
    print("Predictive modeling pipeline completed!")
