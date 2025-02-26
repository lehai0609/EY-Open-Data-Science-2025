'''
Contains functions for creating new features (spatial joins, aggregations, etc.).
'''
#!/usr/bin/env python3
"""
feature_engineering.py

This module prepares the engineered feature dataset by performing spatial joins
and aggregations on the input datasets from data_ingestion and data_processing:
  - UHI target data (df_uhi)
  - Building footprints (gdf_buildings)
  - Sentinel-2 NDVI (gdf_ndvi)
  - Landsat Albedo (gdf_albedo)
  - Weather data (df_weather)
"""

import geopandas as gpd
import numpy as np
import pandas as pd

def create_buffer(df_uhi, buffer_distance=100):
    """Adds a buffer column (geometry) around each UHI point."""
    df = df_uhi.copy()
    df['buffer'] = df.geometry.buffer(buffer_distance)
    return df

def aggregate_building_features(df_uhi, gdf_buildings, buffer_area=31416):
    """
    Performs a spatial join with building footprints and aggregates building attributes.
    Computes advanced metrics including Floor Area Ratio (FAR), Sky View Factor (SVF) 
    approximation, and building density indicators.
    
    Parameters:
        df_uhi: GeoDataFrame with UHI measurement points and buffers
        gdf_buildings: GeoDataFrame with building footprints and attributes
        buffer_area: Area of the buffer in square meters (default: 31416 sq m = 100m radius circle)
        
    Returns:
        df_uhi: GeoDataFrame with building-related features added
    """
    print("Aggregating building density and urban geometry features...")
    
    # Perform spatial join to find buildings that intersect with each buffer
    joined = gpd.sjoin(gdf_buildings, df_uhi.set_geometry('buffer'), how='inner', predicate='intersects')
    
    # Calculate estimated floor count and floor area for each building
    joined['floor_count'] = np.ceil(joined['heightroof'] / 3).clip(1)  # Minimum 1 floor
    joined['floor_area'] = joined['calculated_area_sqm'] * joined['floor_count']
    
    # Group by buffer and calculate metrics
    agg = joined.groupby('index_right').agg(
        # Basic metrics
        building_count=('heightroof', 'count'),
        mean_height=('heightroof', 'mean'),
        total_area=('calculated_area_sqm', 'sum'),
        mean_energy_star=('ENERGY STAR Score', 'mean'),
        
        # Advanced metrics
        max_height=('heightroof', 'max'),
        height_std=('heightroof', 'std'),
        median_height=('heightroof', 'median'),
        
        # Building area statistics
        min_building_area=('calculated_area_sqm', 'min'),
        max_building_area=('calculated_area_sqm', 'max'),
        median_building_area=('calculated_area_sqm', 'median'),
        
        # Total floor area (aggregated from individual building floor areas)
        total_floor_area=('floor_area', 'sum')
    ).reset_index()
    
    # Merge building metrics into UHI dataframe
    df_uhi = df_uhi.reset_index().merge(agg, left_index=True, right_on='index_right', how='left')
    
    # Calculate derived metrics
    
    # Floor Area Ratio (FAR) = Total floor area / buffer area
    df_uhi['floor_area_ratio'] = df_uhi['total_floor_area'] / buffer_area
    
    # Building area ratio (footprint coverage) = Total building footprint area / buffer area
    df_uhi['building_area_ratio'] = df_uhi['total_area'] / buffer_area
    
    # Building density = Number of buildings / buffer area (in hectares)
    df_uhi['building_density'] = df_uhi['building_count'] / (buffer_area / 10000)
    
    # Approximate Sky View Factor (SVF)
    building_coverage = df_uhi['building_area_ratio'].clip(0.01, 0.99)  # Avoid division by zero
    height_factor = df_uhi['mean_height'] / 30  # Normalize heights (assuming 30m is high)
    
    df_uhi['approx_svf'] = 1 - (height_factor * np.sqrt(building_coverage))
    df_uhi['approx_svf'] = df_uhi['approx_svf'].clip(0.1, 1.0)  # Realistic bounds
    
    # Urban canyon metric (H/W ratio - height to width ratio)
    avg_spacing = np.sqrt((buffer_area * (1 - building_coverage)) / df_uhi['building_count'].clip(1))
    df_uhi['canyon_effect'] = (df_uhi['mean_height'] / avg_spacing.clip(1)).clip(0, 5)
    
    # Height-to-area ratio (vertical density)
    df_uhi['height_to_area_ratio'] = df_uhi['mean_height'] / df_uhi['total_area'].clip(1)
    
    # Building regularity (std of height / mean height)
    df_uhi['building_height_regularity'] = (df_uhi['height_std'] / df_uhi['mean_height'].clip(1)).clip(0, 5)
    
    # Create composite urban geometry score (higher = more heat-trapping urban form)
    df_uhi['urban_geometry_score'] = (
        0.3 * (1 - df_uhi['approx_svf']) +  # Lower SVF increases score
        0.3 * df_uhi['building_area_ratio'] +  # Higher coverage increases score
        0.2 * (df_uhi['canyon_effect'] / 5) +  # Higher canyon effect increases score
        0.2 * (df_uhi['mean_height'] / 100)    # Higher buildings increase score
    )
    
    # Fill NaN values for any new columns
    for col in df_uhi.columns:
        if col not in ['geometry', 'buffer'] and df_uhi[col].dtype in ['float64', 'int64']:
            df_uhi[col] = df_uhi[col].fillna(0)
    
    return df_uhi

def aggregate_ndvi_features(df_uhi, gdf_ndvi):
    """
    Joins NDVI pixels with buffered UHI and aggregates vegetation-related statistics.
    Incorporates enhanced vegetation metrics for better quantification of cooling effects.
    """
    print("Aggregating vegetation features from NDVI, EVI, and NDWI data...")
    
    # Perform spatial join
    joined = gpd.sjoin(gdf_ndvi, df_uhi.set_geometry('buffer'), how='inner', predicate='intersects')
    
    # Aggregate vegetation metrics
    agg = joined.groupby('index_right').agg(
        # Basic NDVI statistics
        mean_ndvi=('NDVI', 'mean'),
        median_ndvi=('NDVI', 'median'),
        std_ndvi=('NDVI', 'std'),
        min_ndvi=('NDVI', 'min'),
        max_ndvi=('NDVI', 'max'),
        
        # Enhanced Vegetation Index (EVI) statistics
        mean_evi=('EVI', 'mean'),
        median_evi=('EVI', 'median'),
        
        # Water presence (NDWI) statistics
        mean_ndwi=('NDWI', 'mean')
    ).reset_index()
    # Calculate vegetation class percentages separately
    veg_pct = joined.groupby('index_right').apply(
        lambda x: pd.Series({
            'pct_no_veg': (x['veg_class'] == 'No Vegetation').mean() * 100,
            'pct_low_veg': (x['veg_class'] == 'Low Vegetation').mean() * 100,
            'pct_mod_veg': (x['veg_class'] == 'Moderate Vegetation').mean() * 100,
            'pct_high_veg': (x['veg_class'] == 'High Vegetation').mean() * 100
        })
    ).reset_index()
    # Merge the vegetation percentages with the other aggregated metrics
    agg = agg.merge(veg_pct, on='index_right', how='left')
    
    # NEW: Create feature for sparse vegetation indicator (NDVI < 0.3)
    agg['sparse_veg_area'] = agg['pct_no_veg'] + agg['pct_low_veg']
    
    # NEW: Create vegetation cooling potential score
    # This is a weighted sum where higher vegetation density and coverage contribute more
    agg['veg_cooling_score'] = (
        0.5 * agg['mean_ndvi'] + 
        0.3 * agg['pct_high_veg']/100 + 
        0.2 * agg['mean_evi']
    )
    
    # Merge with UHI dataframe
    df_uhi = df_uhi.merge(agg, left_on='index', right_on='index_right', how='left')
    
    return df_uhi

def aggregate_albedo_features(df_uhi, gdf_albedo, gdf_lst, buffer_size=100):
    """
    Aggregates albedo and LST features for each UHI measurement point.
    
    Parameters:
        df_uhi: GeoDataFrame of UHI measurement points
        gdf_albedo: GeoDataFrame with albedo values
        gdf_lst: GeoDataFrame with Land Surface Temperature (LST) values
        buffer_size: Buffer size in meters for spatial aggregation
        
    Returns:
        DataFrame with UHI points and aggregated albedo and LST features
    """
    print("Aggregating albedo and LST features...")
    
    # Create a copy of the UHI dataframe with reset index to ensure proper joining
    df_uhi_reset = df_uhi.copy().reset_index()
    
    # Create buffers around UHI points for spatial aggregation
    df_uhi_reset['buffer'] = df_uhi_reset.geometry.buffer(buffer_size)
    
    # Create a spatial index on the albedo dataframe to speed up the join
    gdf_albedo_reset = gdf_albedo.copy().reset_index()
    
    # Perform spatial join between UHI buffers and albedo points
    joined_albedo = gpd.sjoin(gdf_albedo_reset, 
                             gpd.GeoDataFrame(df_uhi_reset[['buffer']], 
                                             geometry='buffer', 
                                             crs=df_uhi_reset.crs),
                             predicate='within')
    
    # Debug: Print column names to verify
    print("Columns after albedo join:", joined_albedo.columns.tolist())
    
    # Find the correct join index column - it could be 'index_right' or 'index'
    # depending on how the join was performed
    join_index_col = None
    for col_name in ['index_right', 'index']:
        if col_name in joined_albedo.columns:
            join_index_col = col_name
            break
    
    if join_index_col is None:
        raise ValueError("Could not find a suitable join index column. "
                        f"Available columns: {joined_albedo.columns.tolist()}")
    
    # Group by the join index column and calculate aggregate statistics
    agg_albedo = joined_albedo.groupby(join_index_col).agg(
        mean_albedo=('Albedo', 'mean'),
        min_albedo=('Albedo', 'min'),
        max_albedo=('Albedo', 'max'),
        std_albedo=('Albedo', 'std')
    ).reset_index()
    
    # Reset index on LST dataframe
    gdf_lst_reset = gdf_lst.copy().reset_index()
    
    # Perform spatial join between UHI buffers and LST points
    joined_lst = gpd.sjoin(gdf_lst_reset,
                          gpd.GeoDataFrame(df_uhi_reset[['buffer']], 
                                          geometry='buffer', 
                                          crs=df_uhi_reset.crs),
                          predicate='within')
    
    # Debug: Print column names to verify
    print("Columns after LST join:", joined_lst.columns.tolist())
    
    # Find the correct join index column for LST
    join_index_col_lst = None
    for col_name in ['index_right', 'index']:
        if col_name in joined_lst.columns:
            join_index_col_lst = col_name
            break
    
    if join_index_col_lst is None:
        raise ValueError("Could not find a suitable join index column for LST. "
                        f"Available columns: {joined_lst.columns.tolist()}")
    
    # Group by the join index column and calculate aggregate statistics
    agg_lst = joined_lst.groupby(join_index_col_lst).agg(
        mean_lst=('LST', 'mean'),
        min_lst=('LST', 'min'),
        max_lst=('LST', 'max'),
        std_lst=('LST', 'std')
    ).reset_index()
    
    # Merge albedo features back to UHI points
    df_result = df_uhi_reset.merge(
        agg_albedo, 
        left_on='index', 
        right_on=join_index_col, 
        how='left',
        suffixes=('', '_albedo')
    )
    
    # Drop join index column from albedo if it's not 'index'
    if join_index_col != 'index' and join_index_col in df_result.columns:
        df_result.drop(columns=[join_index_col], inplace=True, errors='ignore')
    
    # Merge LST features
    df_result = df_result.merge(
        agg_lst, 
        left_on='index', 
        right_on=join_index_col_lst, 
        how='left',
        suffixes=('', '_lst')
    )
    
    # Drop unnecessary columns
    df_result.drop(columns=['buffer'], inplace=True, errors='ignore')
    if join_index_col_lst != 'index' and join_index_col_lst in df_result.columns:
        df_result.drop(columns=[join_index_col_lst], inplace=True, errors='ignore')
    
    # Clean up any potential duplicated index columns
    for col in df_result.columns:
        if col.startswith('index_right') or (col.startswith('index_') and col != 'index'):
            df_result.drop(columns=[col], inplace=True, errors='ignore')
    
    print("Albedo and LST feature aggregation complete.")
    return df_result

def integrate_weather_data(df_uhi, df_weather):
    """Matches each UHI record to the closest weather timestamp and merges weather data."""
    df = df_uhi.copy()
    df['weather_time'] = df['datetime'].apply(
        lambda x: df_weather.iloc[(df_weather['Date__Time'] - x).abs().argsort()[0]]['Date__Time']
    )
    df = df.merge(df_weather, left_on='weather_time', right_on='Date__Time', how='left')
    return df

def aggregate_svi_features(df_uhi, gdf_svi, buffer_size=250):
    """
    Aggregates Social Vulnerability Index features for each UHI measurement point.
    
    Parameters:
        df_uhi: GeoDataFrame of UHI measurement points
        gdf_svi: GeoDataFrame with Social Vulnerability Index data
        buffer_size: Buffer size in meters for spatial aggregation
        
    Returns:
        DataFrame with UHI points and aggregated SVI features
    """
    print("Aggregating Social Vulnerability Index features...")
    
    # Create a copy of the UHI dataframe
    df_uhi_copy = df_uhi.copy()
    
    # Handle potential index column conflicts
    if 'level_0' in df_uhi_copy.columns:
        # Rename the existing level_0 column to avoid conflicts
        df_uhi_copy = df_uhi_copy.rename(columns={'level_0': 'original_level_0'})
    
    # Create a new index column instead of using reset_index
    df_uhi_reset = df_uhi_copy.copy()
    df_uhi_reset['temp_join_id'] = range(len(df_uhi_reset))
    
    # Create buffer geometries around UHI points for spatial join
    buffer_geometries = df_uhi_reset.geometry.buffer(buffer_size)
    
    # Create a GeoDataFrame with buffer geometries
    uhi_buffer = gpd.GeoDataFrame(
        df_uhi_reset,
        geometry=buffer_geometries,
        crs=df_uhi_reset.crs
    )
    
    # Perform spatial join between UHI buffers and SVI polygons
    joined = gpd.sjoin(uhi_buffer, gdf_svi, how='left', predicate='intersects')
    
    # Print columns for debugging
    print("Columns after SVI join:", joined.columns.tolist())
    
    # Identify SVI feature columns (excluding geometry and metadata columns)
    svi_columns = [col for col in gdf_svi.columns 
                  if col.startswith('RPL_') and col in joined.columns]
    
    print(f"Found {len(svi_columns)} SVI feature columns to aggregate")
    
    if not svi_columns:
        print("Warning: No SVI feature columns found for aggregation")
        return df_uhi_reset
    
    # Group by the original UHI index and calculate statistics for each SVI metric
    result = df_uhi_reset.copy()
    
    # Perform groupby and aggregation
    if len(joined) > 0:
        agg_dict = {col: ['mean', 'median'] for col in svi_columns}
        agg_svi = joined.groupby('temp_join_id').agg(agg_dict)
        
        # Flatten MultiIndex columns
        agg_svi.columns = ['_'.join(col).strip() for col in agg_svi.columns.values]
        agg_svi = agg_svi.reset_index()
        
        # Merge back to original UHI points
        result = df_uhi_reset.merge(agg_svi, on='temp_join_id', how='left')
    else:
        print("Warning: No matching SVI features found after spatial join")
    
    # Clean up temporary join column
    if 'temp_join_id' in result.columns:
        result.drop(columns=['temp_join_id'], inplace=True)
    
    print("SVI feature aggregation complete.")
    return result

def feature_engineering(df_uhi, gdf_buildings, gdf_ndvi, gdf_albedo, df_weather, gdf_svi, gdf_lst):
    """
    Main feature engineering function.
    
    Inputs:
      - df_uhi: UHI target data (GeoDataFrame)
      - gdf_buildings: Building footprints (GeoDataFrame)
      - gdf_ndvi: Sentinel-2 NDVI data (GeoDataFrame)
      - gdf_albedo: Landsat Albedo data (GeoDataFrame)
      - df_weather: Weather data (DataFrame)
      - gdf_svi: Social Vulnerability Index data (GeoDataFrame)
      - gdf_lst: Landsat Land Surface Temperature data (GeoDataFrame)
    
    Returns:
      - Engineered UHI data with aggregated features (df_uhi)
      - Weather data (unchanged)
    """
    print("Starting feature engineering...")
    
    # 1. Create buffer around UHI points.
    df_uhi = create_buffer(df_uhi, 100)
    
    # 2. Aggregate building, NDVI, and albedo features into df_uhi.
    df_uhi = aggregate_building_features(df_uhi, gdf_buildings, 31416)
    df_uhi = aggregate_ndvi_features(df_uhi, gdf_ndvi)
    df_uhi = aggregate_albedo_features(df_uhi, gdf_albedo, gdf_lst)
    df_uhi = integrate_weather_data(df_uhi, df_weather)
    
    # 3. Integrate SVI data
    df_uhi = aggregate_svi_features(df_uhi, gdf_svi)
    
    print("Feature engineering complete. Sample features from df_uhi:")
    print(df_uhi.head())
    return df_uhi, df_weather
