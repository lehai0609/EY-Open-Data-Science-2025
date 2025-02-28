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
    Trains and evaluates a LightGBM model on the engineered dataset with advanced hyperparameter tuning.
    
    Steps:
      - Excludes non-predictor columns
      - Standardizes numerical features
      - Splits data into training and testing sets
      - Uses RandomizedSearchCV for comprehensive hyperparameter tuning
      - Returns the best model and prints detailed performance metrics
    """
    import numpy as np
    from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
    from sklearn.metrics import r2_score, mean_squared_error
    from sklearn.preprocessing import StandardScaler
    import lightgbm as lgb
    import pandas as pd
    
    print("Starting advanced model training with comprehensive hyperparameter tuning...")
    df_model = df_engineered.copy()
    
    # Exclude non-predictor columns
    exclude = ['index', 'geometry', 'buffer', 'index_right_x', 'index_right_y',
               'index_right0', 'weather_time', 'Date__Time', target_column,
               'datetime', 'Longitude', 'Latitude']
    features = [col for col in df_model.columns if col not in exclude]
    X = df_model[features]
    y = df_model[target_column]
    
    # Clean any potential NaN values in features
    numerical_features = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
    for col in numerical_features:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())
    
    # Standardize numerical features
    scaler = StandardScaler()
    X[numerical_features] = scaler.fit_transform(X[numerical_features])
    
    # Use stratified sampling for more balanced train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Define comprehensive parameter grid for LightGBM
    param_distributions = {
        # Tree structure parameters
        'num_leaves': [31, 50, 70, 90, 120],
        'max_depth': [-1, 5, 7, 9, 11, 15],  # -1 means no limit
        'min_data_in_leaf': [10, 20, 35, 50, 100],
        'min_sum_hessian_in_leaf': [1e-3, 1e-2, 0.1, 1.0],
        
        # Learning parameters
        'learning_rate': [0.01, 0.03, 0.05, 0.07, 0.1, 0.15],
        'n_estimators': [100, 200, 300, 500, 750, 1000],
        'max_bin': [200, 255, 300, 400],
        
        # Regularization parameters
        'lambda_l1': [0, 0.01, 0.05, 0.1, 0.5, 1.0],  # L1 regularization
        'lambda_l2': [0, 0.01, 0.05, 0.1, 0.5, 1.0],  # L2 regularization
        'min_gain_to_split': [0, 0.1, 0.2, 0.5],
        
        # Sampling parameters
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],  # Bagging fraction
        'subsample_freq': [0, 1, 5, 10],  # Bagging frequency
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],  # Feature fraction
    }
    
    # Define base model with core parameters
    base_model = lgb.LGBMRegressor(
        objective='regression',
        metric='rmse',
        n_jobs=-1,
        random_state=42,
        verbose=-1
    )
    
    # Set up cross-validation strategy (5-fold CV)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Use RandomizedSearchCV for efficient hyperparameter exploration
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=100,  # Number of parameter settings sampled
        scoring='r2',
        n_jobs=-1,
        cv=cv,
        verbose=2,
        random_state=42,
        return_train_score=True
    )
    
    # Fit the model
    print("Training model with RandomizedSearchCV (this may take some time)...")
    search.fit(X_train, y_train)
    
    # Get best model
    best_model = search.best_estimator_
    
    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    test_r2 = r2_score(y_test, y_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Print detailed results
    print("\n" + "="*60)
    print("ADVANCED HYPERPARAMETER TUNING RESULTS")
    print("="*60)
    print(f"Best Parameters: {search.best_params_}")
    print(f"Best Cross-Validation R²: {search.best_score_:.6f}")
    print(f"Test R²: {test_r2:.6f}")
    print(f"Test RMSE: {test_rmse:.6f}")
    
    # Calculate and display feature importance
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nTop 15 Most Important Features:")
    print(feature_importance.head(15).to_string(index=False))
    
    # Print hyperparameter search summary
    print("\nHyperparameter Search Summary:")
    print(f"Total parameter combinations evaluated: {len(search.cv_results_['mean_test_score'])}")
    print(f"Mean R² score of all combinations: {np.mean(search.cv_results_['mean_test_score']):.6f}")
    print(f"Standard deviation of R² scores: {np.std(search.cv_results_['mean_test_score']):.6f}")
    
    # Print top 5 parameter sets
    cv_results = pd.DataFrame(search.cv_results_)
    top_results = cv_results.sort_values('mean_test_score', ascending=False).head(5)
    rank_cols = [col for col in top_results.columns if col.startswith('param_')]
    rank_cols.append('mean_test_score')
    rank_cols.append('std_test_score')
    
    print("\nTop 5 Hyperparameter Combinations:")
    pd.set_option('display.max_columns', None)
    print(top_results[rank_cols].to_string(index=False))
    
    return best_model
