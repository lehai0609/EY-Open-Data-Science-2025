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
        mean_ndwi=('NDWI', 'mean'),
        
        # Vegetation class percentages
        pct_no_veg=(lambda x: (x['veg_class'] == 'No Vegetation').mean() * 100),
        pct_low_veg=(lambda x: (x['veg_class'] == 'Low Vegetation').mean() * 100),
        pct_mod_veg=(lambda x: (x['veg_class'] == 'Moderate Vegetation').mean() * 100),
        pct_high_veg=(lambda x: (x['veg_class'] == 'High Vegetation').mean() * 100)
    ).reset_index()
    
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

def aggregate_albedo_features(df_uhi, gdf_albedo, gdf_lst):
    """
    Joins albedo and LST pixels with buffered UHI points to aggregate surface material 
    features and identify heat-retaining surfaces.
    
    Parameters:
        df_uhi: GeoDataFrame with UHI measurement points and buffers
        gdf_albedo: GeoDataFrame with Landsat albedo data
        gdf_lst: GeoDataFrame with Landsat LST data
        
    Returns:
        df_uhi: GeoDataFrame with albedo and LST features added
    """
    print("Aggregating albedo and LST features...")
    
    # Join albedo pixels with buffered UHI
    joined_albedo = gpd.sjoin(gdf_albedo, df_uhi.set_geometry('buffer'), how='inner', predicate='intersects')
    
    # Aggregate albedo statistics
    agg_albedo = joined_albedo.groupby('index_right').agg(
        mean_albedo=('Albedo', 'mean'),
        min_albedo=('Albedo', 'min'),
        max_albedo=('Albedo', 'max'),
        median_albedo=('Albedo', 'median'),
        std_albedo=('Albedo', 'std')
    ).reset_index()
    
    # Merge albedo features into UHI dataframe
    df_uhi = df_uhi.merge(agg_albedo, left_on='index', right_on='index_right', how='left')
    
    # Join LST pixels with buffered UHI
    joined_lst = gpd.sjoin(gdf_lst, df_uhi.set_geometry('buffer'), how='inner', predicate='intersects')
    
    # Aggregate LST statistics
    agg_lst = joined_lst.groupby('index_right').agg(
        mean_lst=('LST', 'mean'),
        min_lst=('LST', 'min'),
        max_lst=('LST', 'max'),
        median_lst=('LST', 'median'),
        std_lst=('LST', 'std')
    ).reset_index()
    
    # Merge LST features into UHI dataframe
    df_uhi = df_uhi.merge(agg_lst, left_on='index', right_on='index_right', how='left')
    
    # Create combined LST-albedo metrics to identify heat-retaining surfaces
    
    # Categorize albedo values
    df_uhi['low_albedo_pct'] = joined_albedo.groupby('index_right').apply(
        lambda x: (x['Albedo'] < 0.2).mean() * 100
    ).reindex(df_uhi['index']).values
    
    df_uhi['medium_albedo_pct'] = joined_albedo.groupby('index_right').apply(
        lambda x: ((x['Albedo'] >= 0.2) & (x['Albedo'] < 0.35)).mean() * 100
    ).reindex(df_uhi['index']).values
    
    df_uhi['high_albedo_pct'] = joined_albedo.groupby('index_right').apply(
        lambda x: (x['Albedo'] >= 0.35).mean() * 100
    ).reindex(df_uhi['index']).values
    
    # Categorize LST values
    q25, q75 = joined_lst['LST'].quantile([0.25, 0.75]).values
    df_uhi['high_lst_pct'] = joined_lst.groupby('index_right').apply(
        lambda x: (x['LST'] > q75).mean() * 100
    ).reindex(df_uhi['index']).values
    
    # Create heat trap indicator (high LST combined with low albedo)
    # For each area, how much of it has both high LST and low albedo
    # First, get the geometrically matching pixels from both datasets
    joined_lst['geometry_wkt'] = joined_lst.geometry.wkt
    joined_albedo['geometry_wkt'] = joined_albedo.geometry.wkt
    
    # Merge on geometry and group index to find pixels that match in both datasets
    joined_both = pd.merge(
        joined_lst[['index_right', 'geometry_wkt', 'LST']],
        joined_albedo[['index_right', 'geometry_wkt', 'Albedo']],
        on=['index_right', 'geometry_wkt']
    )
    
    # Calculate heat trap percentage for each UHI buffer
    df_uhi['heat_trap_pct'] = joined_both.groupby('index_right').apply(
        lambda x: ((x['LST'] > q75) & (x['Albedo'] < 0.2)).mean() * 100
    ).reindex(df_uhi['index']).values
    
    # Create thermal contrast index 
    # Areas with varied surface materials have varied thermal properties
    df_uhi['thermal_contrast'] = df_uhi['std_lst'] * df_uhi['std_albedo']
    
    # Create albedo-LST relationship score
    # Negative correlation between albedo and LST is expected (lower albedo -> higher LST)
    correlation_by_buffer = joined_both.groupby('index_right').apply(
        lambda x: x['LST'].corr(x['Albedo']) if len(x) > 5 else 0
    )
    df_uhi['albedo_lst_correlation'] = correlation_by_buffer.reindex(df_uhi['index']).values
    
    # Create composite heat trap score
    # Higher score = more heat-trapping properties
    df_uhi['heat_trap_score'] = (
        0.4 * (1 - df_uhi['mean_albedo']) +  # Lower albedo increases score
        0.3 * (df_uhi['low_albedo_pct'] / 100) +  # More low-albedo area increases score
        0.3 * (df_uhi['heat_trap_pct'] / 100)  # More heat trap areas increases score
    )
    
    # Fill NaN values for new columns
    for col in df_uhi.columns:
        if col not in ['geometry', 'buffer'] and df_uhi[col].dtype in ['float64', 'int64']:
            df_uhi[col] = df_uhi[col].fillna(0)
    
    return df_uhi

def integrate_weather_data(df_uhi, df_weather):
    """Matches each UHI record to the closest weather timestamp and merges weather data."""
    df = df_uhi.copy()
    df['weather_time'] = df['datetime'].apply(
        lambda x: df_weather.iloc[(df_weather['Date__Time'] - x).abs().argsort()[0]]['Date__Time']
    )
    df = df.merge(df_weather, left_on='weather_time', right_on='Date__Time', how='left')
    return df

def aggregate_svi_features(df_uhi, gdf_svi):
    """
    Performs a spatial join between UHI buffers and SVI data to associate
    social vulnerability metrics with each UHI point.
    
    Parameters:
        df_uhi: GeoDataFrame with UHI measurement points and buffers
        gdf_svi: GeoDataFrame with Social Vulnerability Index data
    
    Returns:
        df_uhi: GeoDataFrame with additional SVI features
    """
    print("Aggregating Social Vulnerability Index features...")
    
    # Create a GeoDataFrame using the 'buffer' column as the active geometry
    uhi_buffer = gpd.GeoDataFrame(df_uhi.copy(), geometry='buffer', crs=df_uhi.crs)
    
    # Perform spatial join to identify which SVI areas intersect with each buffer
    joined = gpd.sjoin(
        gdf_svi,
        uhi_buffer,
        how='inner',
        predicate='intersects',
        lsuffix='_svi',
        rsuffix='_uhi'
    )
    
    # Determine the index column from the join
    if 'index_right' in joined.columns:
        index_col = 'index_right'
    else:
        possible = [col for col in joined.columns if col.startswith('index_right')]
        if possible:
            index_col = possible[0]
        else:
            raise KeyError("No column for the right index was found in the spatial join output.")
    
    # Calculate the area of intersection for weighting
    # This assumes that we want to weight SVI values by their intersection area
    # Get SVI theme columns
    theme_columns = [col for col in joined.columns if col.startswith('RPL_')]
    
    # Aggregate SVI features by UHI buffer, taking the area-weighted mean
    # For simplicity, we'll use unweighted mean here, but area-weighted means could be more accurate
    agg = joined.groupby(index_col).agg({
        theme: 'mean' for theme in theme_columns
    }).reset_index()
    
    # Merge SVI features back into the UHI data
    df_uhi = df_uhi.merge(agg, left_index=True, right_on=index_col, how='left')
    
    # Rename columns to be more descriptive
    rename_dict = {
        'RPL_THEME1': 'svi_socioeconomic',
        'RPL_THEME2': 'svi_household_comp',
        'RPL_THEME3': 'svi_minority_language',
        'RPL_THEME4': 'svi_housing_transport',
        'RPL_THEMES': 'svi_overall'
    }
    
    # Only rename columns that exist
    rename_dict = {k: v for k, v in rename_dict.items() if k in df_uhi.columns}
    df_uhi = df_uhi.rename(columns=rename_dict)
    
    return df_uhi

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
