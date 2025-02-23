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

def create_buffer(df_uhi: gpd.GeoDataFrame, buffer_distance: float = 100) -> gpd.GeoDataFrame:
    """Creates a buffer around each UHI point and returns a new GeoDataFrame."""
    df = df_uhi.copy()
    df['buffer'] = df.geometry.buffer(buffer_distance)
    return df

def spatial_aggregate(target_gdf: gpd.GeoDataFrame, source_gdf: gpd.GeoDataFrame,
                      join_geom: str, predicate: str, agg_dict: dict,
                      target_index: str = 'index') -> pd.DataFrame:
    """
    Performs a spatial join between source and target GeoDataFrames (using target's join_geom column)
    and returns an aggregated DataFrame keyed by the target index.
    """
    target = target_gdf.set_geometry(join_geom)
    joined = gpd.sjoin(source_gdf, target, how='inner', predicate=predicate)
    aggregated = joined.groupby('index_right').agg(agg_dict).reset_index()
    return aggregated

def aggregate_building_features(df_uhi: gpd.GeoDataFrame, gdf_buildings: gpd.GeoDataFrame,
                                buffer_area: float = 31416) -> gpd.GeoDataFrame:
    agg_dict = {
        'heightroof': ['count', 'mean'],
        'calculated_area_sqm': 'sum',
        'ENERGY STAR Score': 'mean'
    }
    building_agg = spatial_aggregate(df_uhi, gdf_buildings, join_geom='buffer', 
                                     predicate='intersects', agg_dict=agg_dict)
    building_agg.columns = ['uhi_index', 'building_count', 'mean_height', 'total_area', 'mean_energy_star']
    df_uhi = df_uhi.reset_index().merge(building_agg, left_on='index', right_on='uhi_index', how='left')
    df_uhi['building_area_ratio'] = df_uhi['total_area'] / buffer_area
    return df_uhi

def integrate_weather_data(df_uhi: pd.DataFrame, df_weather: pd.DataFrame) -> pd.DataFrame:
    df_uhi_sorted = df_uhi.sort_values('datetime')
    df_weather_sorted = df_weather.sort_values('Date__Time')
    df_merged = pd.merge_asof(df_uhi_sorted, df_weather_sorted, left_on='datetime',
                              right_on='Date__Time', direction='nearest')
    return df_merged

def aggregate_socioecon_data(df_person: pd.DataFrame) -> pd.DataFrame:
    return df_person.groupby('CONTROL').agg(
        avg_income=('TOTAL_INC_REC_P', 'mean'),
        max_education=('EDATTAIN_P', 'max')
    ).reset_index()

def enrich_buildings_with_socioecon(gdf_buildings: gpd.GeoDataFrame, df_socioecon: pd.DataFrame) -> gpd.GeoDataFrame:
    gdf_buildings['bin'] = pd.to_numeric(gdf_buildings['bin'], errors='coerce')
    df_socioecon['CONTROL'] = pd.to_numeric(df_socioecon['CONTROL'], errors='coerce')
    enriched = gdf_buildings.merge(df_socioecon, left_on='bin', right_on='CONTROL', how='left')
    enriched['has_socioecon'] = enriched['avg_income'].notnull().astype(int)
    enriched['income_residential_interaction'] = enriched['avg_income'] * enriched.get('is_residential', 1)
    enriched['log_avg_income'] = np.log1p(enriched['avg_income'])
    return enriched

def aggregate_socioecon_features(df_uhi: gpd.GeoDataFrame, enriched_buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    agg_dict = {
        'avg_income': 'mean',
        'max_education': 'max',
        'has_socioecon': 'mean',
        'income_residential_interaction': 'mean',
        'log_avg_income': 'mean'
    }
    socioecon_agg = spatial_aggregate(df_uhi, enriched_buildings, join_geom='buffer', 
                                     predicate='intersects', agg_dict=agg_dict)
    socioecon_agg.columns = ['uhi_index', 'mean_avg_income', 'max_max_education', 
                             'mean_has_socioecon', 'mean_income_interaction', 'mean_log_avg_income']
    df_uhi = df_uhi.reset_index().merge(socioecon_agg, left_on='index', right_on='uhi_index', how='left')
    return df_uhi

def feature_engineering(df_uhi, gdf_buildings, gdf_ndvi, gdf_albedo, df_weather, df_person):
    print("Starting feature engineering...")
    df_uhi = create_buffer(df_uhi, 100)
    df_uhi = aggregate_building_features(df_uhi, gdf_buildings, 31416)
    df_socioecon = aggregate_socioecon_data(df_person)
    enriched_buildings = enrich_buildings_with_socioecon(gdf_buildings, df_socioecon)
    df_uhi = aggregate_socioecon_features(df_uhi, enriched_buildings)
    df_uhi = integrate_weather_data(df_uhi, df_weather)
    print("Feature engineering complete. Sample features from df_uhi:")
    print(df_uhi.head())
    return df_uhi, df_weather
