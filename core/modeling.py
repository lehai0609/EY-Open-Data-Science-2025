'''
Contains functions/classes for training/evaluating models (LightGBM, GridSearch, etc.).
'''
#!/usr/bin/env python3
"""
modelling.py

This module prepares the feature set for modeling and trains a LightGBM model.
It uses the engineered dataset produced by feature_engineering.py.
"""

import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

def prepare_features(df_model, df_weather):
    """
    Prepares the dataset for modeling by imputing missing values, clipping and log-transforming features,
    and handling weather data.
    """
    for col in ['building_count', 'mean_height', 'total_area', 'building_area_ratio']:
        df_model[col] = df_model[col].fillna(0)
    for col in ['mean_ndvi', 'median_ndvi', 'std_ndvi', 'mean_albedo', 'min_albedo', 'max_albedo', 'mean_energy_star']:
        df_model[col] = df_model[col].fillna(df_model[col].mean())
    df_model['mean_ndvi'] = df_model['mean_ndvi'].clip(-1, 1)
    df_model['median_ndvi'] = df_model['median_ndvi'].clip(-1, 1)
    for col in ['total_area', 'building_area_ratio']:
        df_model[f'log_{col}'] = np.log1p(df_model[col])
    weather_cols = [col for col in df_model.columns if col in df_weather.columns and col != 'Date__Time' and not col.startswith('loc_')]
    for col in weather_cols:
        df_model[col] = df_model[col].fillna(df_model[col].mean())
    location_cols = [col for col in df_model.columns if col in ['loc_Bronx', 'loc_Manhattan']]
    for col in location_cols:
        df_model[col] = df_model[col].fillna(df_model[col].mode()[0])
    return df_model

def run_modelling(df_engineered, target_column='UHI Index'):
    """
    Trains and evaluates a LightGBM model on the engineered dataset.
    
    Steps:
      - Excludes non-predictor columns.
      - Standardizes numerical features.
      - Splits data into training and testing sets.
      - Uses GridSearchCV for hyperparameter tuning.
      - Returns the best model and prints performance metrics.
    """
    df_model = df_engineered.copy()
    exclude = ['index', 'geometry', 'buffer', 'index_right_x', 'index_right_y',
               'index_right0', 'weather_time', 'Date__Time', target_column,
               'datetime', 'Longitude', 'Latitude']
    features = [col for col in df_model.columns if col not in exclude]
    X = df_model[features]
    y = df_model[target_column]
    
    numerical_features = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
    scaler = StandardScaler()
    X[numerical_features] = scaler.fit_transform(X[numerical_features])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    lgb_reg = lgb.LGBMRegressor(random_state=42)
    param_grid = {
        'num_leaves': [31, 50],
        'learning_rate': [0.05, 0.1],
        'n_estimators': [100, 200]
    }
    grid = GridSearchCV(lgb_reg, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid.fit(X_train, y_train)
    
    best_model = grid.best_estimator_
    print("\nBest Parameters:", grid.best_params_)
    print("Best Cross-Validation R²:", grid.best_score_)
    
    y_pred = best_model.predict(X_test)
    print("Test R²:", r2_score(y_test, y_pred))
    
    return best_model

# Optional test block
if __name__ == "__main__":
    # Here you would import the engineered features from feature_engineering.py
    # For example:
    # from feature_engineering import feature_engineering
    # Then call run_modelling() with the engineered dataset.
    pass
