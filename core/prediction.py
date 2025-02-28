'''
Contains functions for making predictions on new locations.
'''
#!/usr/bin/env python3
"""
prediction.py

This module handles the prediction workflow for the UHI index:
  - Loads validation data from coordinates
  - Engineers features at new locations using spatial datasets
  - Applies the trained model to make predictions
"""

import os
import pickle
import pandas as pd
import geopandas as gpd
import numpy as np
from core.data_processing import compute_ndvi_sentinel2, compute_landsat_lst_albedo

def load_validation_data(filepath="output/Submission_template_UHI2025-v2.csv"):
    """
    Loads validation data from CSV file and prepares it for feature engineering.
    
    Parameters:
        filepath: Path to validation CSV with coordinates
        
    Returns:
        GeoDataFrame with points and buffer for spatial analysis
    """
    print(f"Loading validation data from: {filepath}")
    df = pd.read_csv(filepath)
    
    # Keep original columns as-is
    original_df = df.copy()
    
    # Convert to GeoDataFrame with proper projection
    gdf = gpd.GeoDataFrame(
        original_df,
        geometry=gpd.points_from_xy(original_df.Longitude, original_df.Latitude),
        crs="EPSG:4326"
    )
    gdf = gdf.to_crs("EPSG:2263")  # NYC local projection
    
    # Add datetime column (using same time as training data)
    gdf['datetime'] = pd.to_datetime('24-07-2021 15:30', format="%d-%m-%Y %H:%M")
    
    # Create buffer for spatial analysis (100m)
    gdf['buffer'] = gdf.geometry.buffer(100)
    
    print(f"Loaded {len(gdf)} validation points")
    return gdf

def engineer_validation_features(validation_gdf, gdf_buildings, gdf_ndvi, 
                                gdf_albedo, df_weather, gdf_svi, gdf_lst):
    """
    Engineers features for validation data points using spatial data.
    """
    print("Starting feature engineering for validation points...")
    print(f"Input validation points: {len(validation_gdf)}")
    
    # Add feature engineering steps with verification
    validation_gdf = aggregate_building_features_for_prediction(validation_gdf, gdf_buildings)
    print(f"After building features: {len(validation_gdf)} points")
    
    validation_gdf = aggregate_ndvi_features_for_prediction(validation_gdf, gdf_ndvi)
    print(f"After NDVI features: {len(validation_gdf)} points")
    
    validation_gdf = aggregate_albedo_features_for_prediction(validation_gdf, gdf_albedo, gdf_lst)
    print(f"After albedo features: {len(validation_gdf)} points")
    
    validation_gdf = integrate_weather_data_for_prediction(validation_gdf, df_weather)
    print(f"After weather integration: {len(validation_gdf)} points")
    
    validation_gdf = aggregate_svi_features_for_prediction(validation_gdf, gdf_svi)
    print(f"After SVI features: {len(validation_gdf)} points")
    
    # Final check for duplicates
    if len(validation_gdf) != len(validation_gdf.drop_duplicates(['Longitude', 'Latitude'])):
        print("WARNING: Duplicate points detected, removing duplicates...")
        validation_gdf = validation_gdf.drop_duplicates(['Longitude', 'Latitude'])
        
    print(f"Final feature engineering complete: {len(validation_gdf)} points")
    return validation_gdf

def aggregate_building_features_for_prediction(df_validation, gdf_buildings, buffer_area=31416):
    """
    Performs spatial join with building footprints and aggregates urban geometry metrics.
    Adapted from feature_engineering.py for prediction workflow.
    """
    # Perform spatial join to find buildings that intersect with each buffer
    joined = gpd.sjoin(gdf_buildings, df_validation.set_geometry('buffer'), 
                       how='inner', predicate='intersects')
    
    # Calculate estimated floor count and floor area
    joined['floor_count'] = np.ceil(joined['heightroof'] / 3).clip(1)
    joined['floor_area'] = joined['calculated_area_sqm'] * joined['floor_count']
    
    # Group by buffer and calculate metrics
    agg = joined.groupby('index_right').agg(
        building_count=('heightroof', 'count'),
        mean_height=('heightroof', 'mean'),
        total_area=('calculated_area_sqm', 'sum'),
        mean_energy_star=('ENERGY STAR Score', 'mean'),
        max_height=('heightroof', 'max'),
        height_std=('heightroof', 'std'),
        median_height=('heightroof', 'median'),
        min_building_area=('calculated_area_sqm', 'min'),
        max_building_area=('calculated_area_sqm', 'max'),
        median_building_area=('calculated_area_sqm', 'median'),
        total_floor_area=('floor_area', 'sum')
    ).reset_index()
    
    # Merge building metrics into validation dataframe
    df_result = df_validation.reset_index().merge(
        agg, left_index=True, right_on='index_right', how='left'
    )
    
    # Calculate derived urban form metrics
    df_result['floor_area_ratio'] = df_result['total_floor_area'] / buffer_area
    df_result['building_area_ratio'] = df_result['total_area'] / buffer_area
    df_result['building_density'] = df_result['building_count'] / (buffer_area / 10000)
    
    # Approximate Sky View Factor (SVF)
    building_coverage = df_result['building_area_ratio'].clip(0.01, 0.99)
    height_factor = df_result['mean_height'] / 30
    df_result['approx_svf'] = 1 - (height_factor * np.sqrt(building_coverage))
    df_result['approx_svf'] = df_result['approx_svf'].clip(0.1, 1.0)
    
    # Urban canyon effect
    avg_spacing = np.sqrt((buffer_area * (1 - building_coverage)) / df_result['building_count'].clip(1))
    df_result['canyon_effect'] = (df_result['mean_height'] / avg_spacing.clip(1)).clip(0, 5)
    
    # Height-to-area ratio and building regularity
    df_result['height_to_area_ratio'] = df_result['mean_height'] / df_result['total_area'].clip(1)
    df_result['building_height_regularity'] = (df_result['height_std'] / df_result['mean_height'].clip(1)).clip(0, 5)
    
    # Urban geometry score (composite indicator)
    df_result['urban_geometry_score'] = (
        0.3 * (1 - df_result['approx_svf']) +
        0.3 * df_result['building_area_ratio'] +
        0.2 * (df_result['canyon_effect'] / 5) +
        0.2 * (df_result['mean_height'] / 100)
    )
    
    # Fill NaN values
    for col in df_result.columns:
        if col not in ['geometry', 'buffer'] and df_result[col].dtype in ['float64', 'int64']:
            df_result[col] = df_result[col].fillna(0)
    
    return df_result

def aggregate_ndvi_features_for_prediction(df_validation, gdf_ndvi):
    """
    Joins NDVI data with validation points and aggregates vegetation metrics.
    Adapted from feature_engineering.py for prediction workflow.
    """
    # Perform spatial join
    joined = gpd.sjoin(gdf_ndvi, df_validation.set_geometry('buffer'), 
                      how='inner', predicate='intersects')
    
    # Aggregate vegetation metrics
    agg = joined.groupby('index_right').agg(
        mean_ndvi=('NDVI', 'mean'),
        median_ndvi=('NDVI', 'median'),
        std_ndvi=('NDVI', 'std'),
        min_ndvi=('NDVI', 'min'),
        max_ndvi=('NDVI', 'max'),
        mean_evi=('EVI', 'mean'),
        median_evi=('EVI', 'median'),
        mean_ndwi=('NDWI', 'mean')
    ).reset_index()
    
    # Calculate vegetation class percentages
    veg_pct = joined.groupby('index_right').apply(
        lambda x: pd.Series({
            'pct_no_veg': (x['veg_class'] == 'No Vegetation').mean() * 100,
            'pct_low_veg': (x['veg_class'] == 'Low Vegetation').mean() * 100,
            'pct_mod_veg': (x['veg_class'] == 'Moderate Vegetation').mean() * 100,
            'pct_high_veg': (x['veg_class'] == 'High Vegetation').mean() * 100
        })
    ).reset_index()
    
    # Merge the vegetation metrics
    agg = agg.merge(veg_pct, on='index_right', how='left')
    
    # Create derived vegetation features
    agg['sparse_veg_area'] = agg['pct_no_veg'] + agg['pct_low_veg']
    agg['veg_cooling_score'] = (
        0.5 * agg['mean_ndvi'] + 
        0.3 * agg['pct_high_veg']/100 + 
        0.2 * agg['mean_evi']
    )
    
    # Merge with validation dataframe
    df_result = df_validation.merge(agg, left_on='index', right_on='index_right', how='left')
    
    # Fill NaN values
    for col in agg.columns:
        if col != 'index_right' and col in df_result.columns and df_result[col].dtype in ['float64', 'int64']:
            df_result[col] = df_result[col].fillna(0)
            
    return df_result

def aggregate_albedo_features_for_prediction(df_validation, gdf_albedo, gdf_lst):
    """
    Aggregates albedo and land surface temperature features for prediction.
    Adapted from feature_engineering.py for prediction workflow.
    """
    # Create copies with reset indexes to avoid join issues
    df_validation_reset = df_validation.copy().reset_index(drop=True)
    df_validation_reset['point_id'] = range(len(df_validation_reset))
    gdf_albedo_reset = gdf_albedo.copy().reset_index()
    
    # Perform spatial join for albedo
    joined_albedo = gpd.sjoin(
        gdf_albedo_reset,
        gpd.GeoDataFrame(df_validation_reset[['point_id', 'buffer']], 
                        geometry='buffer', crs=df_validation_reset.crs),
        predicate='within'
    )
    
    # Calculate albedo statistics
    agg_albedo = joined_albedo.groupby('point_id').agg(
        mean_albedo=('Albedo', 'mean'),
        min_albedo=('Albedo', 'min'),
        max_albedo=('Albedo', 'max'),
        std_albedo=('Albedo', 'std')
    ).reset_index()
    
    # Perform spatial join for LST
    gdf_lst_reset = gdf_lst.copy().reset_index()
    joined_lst = gpd.sjoin(
        gdf_lst_reset,
        gpd.GeoDataFrame(df_validation_reset[['point_id', 'buffer']], 
                        geometry='buffer', crs=df_validation_reset.crs),
        predicate='within'
    )
    
    # Calculate LST statistics
    agg_lst = joined_lst.groupby('point_id').agg(
        mean_lst=('LST', 'mean'),
        min_lst=('LST', 'min'),
        max_lst=('LST', 'max'),
        std_lst=('LST', 'std')
    ).reset_index()
    
    # Merge albedo and LST features back to validation points
    df_result = df_validation_reset.merge(agg_albedo, on='point_id', how='left')
    df_result = df_result.merge(agg_lst, on='point_id', how='left')
    
    # Fill NaN values
    albedo_cols = [col for col in agg_albedo.columns if col != 'point_id']
    lst_cols = [col for col in agg_lst.columns if col != 'point_id']
    
    for col in albedo_cols + lst_cols:
        if col in df_result.columns:
            df_result[col] = df_result[col].fillna(df_result[col].median() if not df_result[col].isnull().all() else 0)
    
    # Drop temporary columns
    if 'point_id' in df_result.columns:
        df_result.drop(columns=['point_id'], inplace=True)
        
    return df_result

def integrate_weather_data_for_prediction(df_validation, df_weather):
    """
    Matches each validation point to the closest weather timestamp.
    Adapted from feature_engineering.py for prediction workflow.
    """
    df_result = df_validation.copy()
    
    # Find the closest weather timestamp for each validation point
    df_result['weather_time'] = df_result['datetime'].apply(
        lambda x: df_weather.iloc[(df_weather['Date__Time'] - x).abs().argsort()[0]]['Date__Time']
    )
    
    # Merge weather data
    df_result = df_result.merge(df_weather, left_on='weather_time', right_on='Date__Time', how='left')
    
    return df_result

def aggregate_svi_features_for_prediction(df_validation, gdf_svi, buffer_size=250):
    """
    Aggregates Social Vulnerability Index features for validation points.
    Adapted from feature_engineering.py for prediction workflow.
    """
    # Create a unique ID for each validation point
    df_validation_reset = df_validation.copy()
    df_validation_reset['svi_join_id'] = range(len(df_validation_reset))
    
    # Create buffer geometries for SVI analysis (larger than UHI buffer)
    buffer_geometries = df_validation_reset.geometry.buffer(buffer_size)
    
    # Create a GeoDataFrame with the buffers
    validation_buffer = gpd.GeoDataFrame(
        df_validation_reset[['svi_join_id']],
        geometry=buffer_geometries,
        crs=df_validation_reset.crs
    )
    
    # Perform spatial join with SVI data
    joined = gpd.sjoin(validation_buffer, gdf_svi, how='left', predicate='intersects')
    
    # Identify SVI feature columns
    svi_columns = [col for col in gdf_svi.columns 
                  if col.startswith('RPL_') and col in joined.columns]
    
    # Perform aggregation if SVI features are found
    if len(joined) > 0 and len(svi_columns) > 0:
        agg_dict = {col: ['mean', 'median'] for col in svi_columns}
        agg_svi = joined.groupby('svi_join_id').agg(agg_dict)
        
        # Flatten MultiIndex columns
        agg_svi.columns = ['_'.join(col).strip() for col in agg_svi.columns.values]
        agg_svi = agg_svi.reset_index()
        
        # Merge back to validation points
        df_result = df_validation_reset.merge(agg_svi, on='svi_join_id', how='left')
    else:
        print("Warning: No SVI features found or no matching SVI data for validation points")
        df_result = df_validation_reset
    
    # Drop temporary join column
    if 'svi_join_id' in df_result.columns:
        df_result.drop(columns=['svi_join_id'], inplace=True)
    
    return df_result

def prepare_features_for_prediction(df_with_features, model):
    """
    Prepares feature data for prediction, ensuring compatibility with the model.
    
    Parameters:
        df_with_features: DataFrame with engineered features
        model: Trained model with feature_name_ attribute
        
    Returns:
        Feature matrix ready for prediction
    """
    from sklearn.preprocessing import StandardScaler
    
    df_model = df_with_features.copy()
    
    # Handle missing values in important features
    for col in ['building_count', 'mean_height', 'total_area', 'building_area_ratio']:
        if col in df_model.columns:
            df_model[col] = df_model[col].fillna(0)
    
    for col in ['mean_ndvi', 'median_ndvi', 'mean_albedo', 'mean_energy_star']:
        if col in df_model.columns:
            df_model[col] = df_model[col].fillna(df_model[col].mean() if not df_model[col].isnull().all() else 0)
    
    # Create log transformations for skewed features
    for col in ['total_area', 'building_area_ratio']:
        if col in df_model.columns:
            df_model[f'log_{col}'] = np.log1p(df_model[col])
    
    # Get model features
    model_features = model.feature_name_
    
    # Check for missing features
    missing_features = [feat for feat in model_features if feat not in df_model.columns]
    if missing_features:
        print(f"Adding {len(missing_features)} missing features required by the model")
        for feat in missing_features:
            df_model[feat] = 0
    
    # Select only features used by the model
    X = df_model[model_features]
    
    # Ensure all numeric features have valid values
    for col in X.columns:
        if X[col].dtype in ['float64', 'int64'] and X[col].isnull().any():
            X[col] = X[col].fillna(X[col].median() if not X[col].isnull().all() else 0)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled

def load_trained_model(model_path="output/best_model.pkl"):
    """
    Loads the trained model from disk.
    
    Parameters:
        model_path: Path to the saved model file
        
    Returns:
        Trained model object
    """
    print(f"Loading trained model from: {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def make_predictions(validation_gdf, model):
    """
    Makes UHI Index predictions using the trained model.
    
    Parameters:
        validation_gdf: GeoDataFrame with engineered features
        model: Trained model
        
    Returns:
        GeoDataFrame with predictions added
    """
    print("Preparing features for prediction...")
    X_scaled = prepare_features_for_prediction(validation_gdf, model)
    
    print("Making predictions...")
    predictions = model.predict(X_scaled)
    
    # Add predictions to the GeoDataFrame
    result_gdf = validation_gdf.copy()
    result_gdf['UHI_Index'] = predictions
    
    print(f"Generated predictions for {len(result_gdf)} points")
    return result_gdf

def save_predictions(gdf_with_predictions, output_path="output/UHI_predictions.csv"):
    """
    Saves predictions to a CSV file, preserving original coordinates.
    
    Parameters:
        gdf_with_predictions: GeoDataFrame with predictions
        output_path: Path to save the output CSV
    """
    # Prepare output dataframe with required columns, using original coordinates
    output_df = pd.DataFrame({
        'Longitude': gdf_with_predictions['Longitude'],  # Use original column instead of geometry.x
        'Latitude': gdf_with_predictions['Latitude'],    # Use original column instead of geometry.y
        'UHI Index': gdf_with_predictions['UHI_Index']   # Match the expected column name
    })
    
    # Remove any potential duplicates
    output_df = output_df.drop_duplicates(['Longitude', 'Latitude'])
    
    # Verify the row count matches the expected count
    print(f"Number of rows in output file: {len(output_df)}")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    output_df.to_csv(output_path, index=False)
    print(f"Predictions saved to: {output_path}")

def run_prediction_pipeline(validation_file_path, model_path, output_path):
    """
    Orchestrates the full prediction workflow.
    
    Parameters:
        validation_file_path: Path to validation CSV with coordinates
        model_path: Path to saved model
        output_path: Path to save predictions
    """
    from core.data_ingestion import (
        read_building_footprints,
        read_energy_star_data,
        read_weather_data,
        read_social_vulnerability_index
    )
    
    # Step 1: Load validation data
    validation_gdf = load_validation_data(validation_file_path)
    
    # Step 2: Load required spatial datasets
    print("Loading spatial datasets for feature engineering...")
    gdf_buildings = read_building_footprints()
    df_energy_star = read_energy_star_data()
    gdf_buildings = gdf_buildings.merge(df_energy_star, on='bin', how='left')
    df_weather = read_weather_data()
    gdf_svi = read_social_vulnerability_index()
    
    # Step 3: Process remote sensing data
    print("Processing remote sensing data...")
    gdf_ndvi = compute_ndvi_sentinel2()
    gdf_lst, gdf_albedo = compute_landsat_lst_albedo()
    
    # Step 4: Engineer features for validation points
    validation_with_features = engineer_validation_features(
        validation_gdf, gdf_buildings, gdf_ndvi, gdf_albedo, 
        df_weather, gdf_svi, gdf_lst
    )
    
    # Step 5: Load trained model
    model = load_trained_model(model_path)
    
    # Step 6: Make predictions
    validation_with_predictions = make_predictions(validation_with_features, model)
    
    # Step 7: Save predictions
    save_predictions(validation_with_predictions, output_path)
    
    print("Prediction pipeline completed successfully!")