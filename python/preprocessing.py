"""
feature_engineering.py
NFL Veterans Team Change Analysis
Purpose: Advanced ML feature engineering and predictive modeling preparation

This script creates features for:
1. Predicting which veterans will succeed after team changes
2. Career trajectory modeling
3. Propensity score matching for causal inference
4. Time-series forecasting of performance decline
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import zscore
import warnings
warnings.filterwarnings('ignore')

# Directories
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
ML_DIR = Path("data/ml_features")
ML_DIR.mkdir(parents=True, exist_ok=True)

print("="
