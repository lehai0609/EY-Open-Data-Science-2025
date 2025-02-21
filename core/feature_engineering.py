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

def feature_engineering(df_uhi, gdf_buildings, gdf_ndvi, gdf_albedo, df_weather):
    """
    Main feature engineering function.
    
    Inputs:
      - df_uhi: UHI target data (GeoDataFrame)
      - gdf_buildings: Building footprints (GeoDataFrame)
      - gdf_ndvi: Sentinel-2 NDVI data (GeoDataFrame)
      - gdf_albedo: Landsat Albedo data (GeoDataFrame)
      - df_weather: Weather data (DataFrame)
    
    Returns:
      - Engineered UHI data with aggregated features
      - Weather data (unchanged)
    """
    print("Starting feature engineering...")
    df_uhi = create_buffer(df_uhi, 100)
    df_uhi = aggregate_building_features(df_uhi, gdf_buildings, 31416)
    df_uhi = aggregate_ndvi_features(df_uhi, gdf_ndvi)
    df_uhi = aggregate_albedo_features(df_uhi, gdf_albedo)
    df_uhi = integrate_weather_data(df_uhi, df_weather)
    print("Feature engineering complete. Sample features:")
    print(df_uhi.head())
    return df_uhi, df_weather

# Optional test block
if __name__ == "__main__":
    # Here you would import your previously processed data
    # For example:
    # from data_ingestion import read_target_variables, read_building_footprints, read_weather_data
    # from data_processing import compute_ndvi_sentinel2, compute_landsat_lst_albedo
    # Then call feature_engineering() with these inputs.
    pass
