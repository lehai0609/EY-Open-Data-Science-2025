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
    """Performs a spatial join with building footprints and aggregates building attributes."""
    joined = gpd.sjoin(gdf_buildings, df_uhi.set_geometry('buffer'), how='inner', predicate='intersects')
    agg = joined.groupby('index_right').agg(
        building_count=('heightroof', 'count'),
        mean_height=('heightroof', 'mean'),
        total_area=('calculated_area_sqm', 'sum'),
        mean_energy_star=('ENERGY STAR Score', 'mean')
    ).reset_index()
    df_uhi = df_uhi.reset_index().merge(agg, left_index=True, right_on='index_right', how='left')
    df_uhi['building_area_ratio'] = df_uhi['total_area'] / buffer_area
    return df_uhi

def aggregate_ndvi_features(df_uhi, gdf_ndvi):
    """Joins NDVI pixels with buffered UHI and aggregates NDVI statistics."""
    joined = gpd.sjoin(gdf_ndvi, df_uhi.set_geometry('buffer'), how='inner', predicate='intersects')
    agg = joined.groupby('index_right').agg(
        mean_ndvi=('NDVI', 'mean'),
        median_ndvi=('NDVI', 'median'),
        std_ndvi=('NDVI', 'std')
    ).reset_index()
    df_uhi = df_uhi.merge(agg, left_on='index', right_on='index_right', how='left')
    return df_uhi

def aggregate_albedo_features(df_uhi, gdf_albedo):
    """Joins albedo pixels with buffered UHI and aggregates albedo statistics."""
    joined = gpd.sjoin(gdf_albedo, df_uhi.set_geometry('buffer'), how='inner', predicate='intersects')
    agg = joined.groupby('index_right0').agg(
        mean_albedo=('Albedo', 'mean'),
        min_albedo=('Albedo', 'min'),
        max_albedo=('Albedo', 'max')
    ).reset_index()
    df_uhi = df_uhi.merge(agg, left_on='index', right_on='index_right0', how='left')
    return df_uhi

def integrate_weather_data(df_uhi, df_weather):
    """Matches each UHI record to the closest weather timestamp and merges weather data."""
    df = df_uhi.copy()
    df['weather_time'] = df['datetime'].apply(
        lambda x: df_weather.iloc[(df_weather['Date__Time'] - x).abs().argsort()[0]]['Date__Time']
    )
    df = df.merge(df_weather, left_on='weather_time', right_on='Date__Time', how='left')
    return df

def aggregate_socioecon_data(df_person):
    """
    Aggregates socio-economic data by housing unit (CONTROL code):
      - Computes average housing income.
      - Finds the highest tenant education level.
    Returns a DataFrame keyed by CONTROL.
    """
    agg_df = df_person.groupby('CONTROL').agg(
        avg_income=('TOTAL_INC_REC_P', 'mean'),
        max_education=('EDATTAIN_P', 'max')
    ).reset_index()
    return agg_df

def aggregate_socioecon_features(df_uhi, gdf_buildings):
    """
    Joins the updated building dataset (which includes socio-economic features)
    with df_uhi based on the UHI buffer. Aggregates socio-economic features
    (e.g., mean income, highest education) into df_uhi.
    """
    import geopandas as gpd

    # Reset indices to ensure they are simple sequential integers.
    df_uhi = df_uhi.reset_index(drop=True)
    gdf_buildings = gdf_buildings.reset_index(drop=True)
    
    # Create a GeoDataFrame using the 'buffer' column as the active geometry.
    uhi_buffer = gpd.GeoDataFrame(df_uhi.copy(), geometry='buffer', crs=df_uhi.crs)
    
    # Perform the spatial join.
    joined = gpd.sjoin(
        gdf_buildings,
        uhi_buffer,
        how='inner',
        predicate='intersects',
        lsuffix='_build',
        rsuffix='_uhi'
    )
    
    # Determine which column contains the right index.
    # sjoin should create an "index_right" column, but if not, pick one that starts with it.
    if 'index_right' in joined.columns:
        index_col = 'index_right'
    else:
        possible = [col for col in joined.columns if col.startswith('index_right')]
        if possible:
            index_col = possible[0]
        else:
            raise KeyError("No column for the right index was found in the spatial join output.")

    # Aggregate socio-economic features by the UHI index.
    agg = joined.groupby(index_col).agg(
        mean_avg_income=('avg_income', 'mean'),
        max_max_education=('max_education', 'max'),
        mean_has_socioecon=('has_socioecon', 'mean'),
        mean_income_interaction=('income_residential_interaction', 'mean'),
        mean_log_avg_income=('log_avg_income', 'mean')
    ).reset_index()
    
    # Merge the aggregated features back into df_uhi.
    df_uhi = df_uhi.merge(agg, left_index=True, right_on=index_col, how='left')
    return df_uhi


def feature_engineering(df_uhi, gdf_buildings, gdf_ndvi, gdf_albedo, df_weather, df_person):
    """
    Main feature engineering function.
    
    Inputs:
      - df_uhi: UHI target data (GeoDataFrame)
      - gdf_buildings: Building footprints (GeoDataFrame)
      - gdf_ndvi: Sentinel-2 NDVI data (GeoDataFrame)
      - gdf_albedo: Landsat Albedo data (GeoDataFrame)
      - df_weather: Weather data (DataFrame)
      - df_person: Socio-economic data from person_puf_21.csv (DataFrame)
    
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
    df_uhi = aggregate_albedo_features(df_uhi, gdf_albedo)
    df_uhi = integrate_weather_data(df_uhi, df_weather)
    
    # 3. --- Socio-Economic Data Integration ---
    # a. Aggregate the person-level socio-economic data.
    agg_socioecon = aggregate_socioecon_data(df_person)
    
    # b. Merge the aggregated socio-economic data into the building dataset.
    #    Here, we assume that the building dataset has a 'bin' column that corresponds to the CONTROL code.
    # Convert the 'bin' column to numeric to match the type of 'CONTROL'
    gdf_buildings['bin'] = pd.to_numeric(gdf_buildings['bin'], errors='coerce')
    agg_socioecon['CONTROL'] = pd.to_numeric(agg_socioecon['CONTROL'], errors='coerce')
    gdf_buildings = gdf_buildings.merge(agg_socioecon, left_on='bin', right_on='CONTROL', how='left')
    
    # c. Create new socio-economic features in the building dataset.
    # Indicator: 1 if socio-economic data exists, 0 otherwise.
    gdf_buildings['has_socioecon'] = gdf_buildings['avg_income'].notnull().astype(int)
    
    # Example: Define an 'is_residential' flag (replace with actual logic if available).
    if 'is_residential' not in gdf_buildings.columns:
        gdf_buildings['is_residential'] = 1
    gdf_buildings['income_residential_interaction'] = gdf_buildings['avg_income'] * gdf_buildings['is_residential']
    gdf_buildings['log_avg_income'] = np.log1p(gdf_buildings['avg_income'])
    
    # d. Aggregate the socio-economic features from the building dataset into df_uhi.
    df_uhi = aggregate_socioecon_features(df_uhi, gdf_buildings)
    
    print("Feature engineering complete. Sample features from df_uhi:")
    print(df_uhi.head())
    return df_uhi, df_weather
