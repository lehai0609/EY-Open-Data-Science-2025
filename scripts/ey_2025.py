#!/usr/bin/env python3
"""
Combined Refactored Script for EY Open Data Science 2025 Project

This script consolidates all notebooks from the 'scripts' folder into one modular file.
It includes:
  - Reading target variables
  - Processing building footprints, Sentinel-2, Landsat, and weather data
  - Feature engineering (including spatial joins and aggregations)
  - Updated model training using LightGBM with hyperparameter tuning, with proper feature imputation and transformation
"""

import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import xarray as xr
import rioxarray as rio
import rasterio
from shapely.geometry import Point
import stackstac
import pystac_client
import planetary_computer
from odc.stac import stac_load
import pyarrow.parquet as pq
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score
import lightgbm as lgb

# ----------------------------------------------------------------------
# 01 Reading Target Variables
# ----------------------------------------------------------------------
def read_target_variables():
    print("Reading UHI target variables...")
    file_path = "../input/Training_data_uhi_index.csv"
    df_uhi = pd.read_csv(file_path)
    df_uhi['datetime'] = pd.to_datetime(df_uhi['datetime'], format="%d-%m-%Y %H:%M")
    print("Sample of UHI data:")
    print(df_uhi.head())
    # Convert to GeoDataFrame and reproject to EPSG:2263
    df_uhi_geo = gpd.GeoDataFrame(
        df_uhi,
        geometry=gpd.points_from_xy(df_uhi.Longitude, df_uhi.Latitude),
        crs="EPSG:4326"
    )
    df_uhi_geo = df_uhi_geo.to_crs("EPSG:2263")
    output_path = '../output/uhi_data_processed.parquet'
    df_uhi_geo.to_parquet(output_path)
    print(f"UHI data saved to: {output_path}")
    return df_uhi_geo

# ----------------------------------------------------------------------
# 02 Processing Building Footprints
# ----------------------------------------------------------------------
def process_building_footprints():
    print("Processing building footprints...")
    file_path = "../input/Building Footprints_20250131.geojson"
    gdf_buildings = gpd.read_file(file_path)
    
    cols_to_convert = ['shape_area', 'heightroof', 'cnstrct_yr', 'groundelev']
    for col in cols_to_convert:
        gdf_buildings[col] = pd.to_numeric(gdf_buildings[col], errors='coerce')
    gdf_buildings['lstmoddate'] = pd.to_datetime(gdf_buildings['lstmoddate'], errors='coerce')
    
    gdf_buildings = gdf_buildings.to_crs(epsg=2263)
    gdf_buildings['calculated_area_sqft'] = gdf_buildings.geometry.area
    gdf_buildings['calculated_area_sqm'] = gdf_buildings['calculated_area_sqft'] * 0.092903
    
    columns_to_remove = ['name', 'base_bbl', 'mpluto_bbl', 'cnstrct_yr', 'doitt_id',
                         'geomsource', 'lststatype', 'shape_len', 'globalid',
                         'feat_code', 'lstmoddate', 'calculated_area_sqft']
    gdf_buildings.drop(columns=columns_to_remove, inplace=True, errors='ignore')
    
    median_height = gdf_buildings['heightroof'].median()
    gdf_buildings['heightroof'] = gdf_buildings['heightroof'].fillna(median_height)
    median_elevation = gdf_buildings['groundelev'].median()
    gdf_buildings['groundelev'] = gdf_buildings['groundelev'].fillna(median_elevation)

    # ---- New code for ENERGY STAR integration ----
    # Load the ENERGY STAR Rating dataset (assumed to be a shapefile)
    energy_star_file = "../input/NYC_Building_Energy_20250216.csv"  # update path as needed
    gdf_energy_star = pd.read_csv(energy_star_file)
    
    # Select and rename columns to ensure a common join key; assuming the rating is in 'ENERGY_STAR_RATING'
    gdf_energy_star = gdf_energy_star[['NYC Building Identification Number (BIN)', 'ENERGY STAR Score']]
    gdf_energy_star = gdf_energy_star.rename(columns={'NYC Building Identification Number (BIN)': 'bin'})
    
    # Ensure the ENERGY STAR dataset is in the same CRS
    # gdf_energy_star = gdf_energy_star.to_crs(epsg=2263)
    
    # Merge the ENERGY STAR Rating into the building footprints using the 'bin' field
    gdf_buildings = gdf_buildings.merge(gdf_energy_star, on='bin', how='left')
    # ------------------------------------------------
    
    output_path = "../output/building_data_processed.parquet"
    gdf_buildings.to_parquet(output_path, index=False)
    print(f"Building data saved to: {output_path}")
    return gdf_buildings

# ----------------------------------------------------------------------
# 03 Processing Sentinel-2 Data (NDVI)
# ----------------------------------------------------------------------
def process_sentinel2_data():
    print("Processing Sentinel-2 data for NDVI...")
    warnings.filterwarnings('ignore')
    lower_left = (40.75, -74.01)
    upper_right = (40.88, -73.86)
    bounds = (lower_left[1], lower_left[0], upper_right[1], upper_right[0])
    time_window = "2021-07-01/2021-08-01"
    
    stac = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    search = stac.search(
        bbox=bounds,
        datetime=time_window,
        collections=["sentinel-2-l2a"],
        query={"eo:cloud_cover": {"lt": 30}},
    )
    items = list(search.get_items())
    print("Number of Sentinel-2 scenes found:", len(items))
    
    resolution = 10
    data = stac_load(
        items,
        bands=["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12", "SCL"],
        crs="EPSG:2263",
        resolution=resolution,
        chunks={"x": 2048, "y": 2048},
        dtype="uint16",
        patch_url=planetary_computer.sign,
        bbox=bounds
    )
    
    valid_mask = (data.SCL == 4) | (data.SCL == 5)
    ndvi = (data.B08 - data.B04) / (data.B08 + data.B04 + 1e-6)
    ndvi = ndvi.where(valid_mask, other=np.nan)
    ndvi = ndvi.compute()
    
    target_date = np.datetime64("2021-07-24")
    time_diffs = abs(data.time - target_date)
    closest_time_index = int(time_diffs.argmin())
    ndvi_slice = ndvi.isel(time=closest_time_index)
    
    ndvi_df = ndvi_slice.to_dataframe(name='NDVI').reset_index()
    gdf_ndvi = gpd.GeoDataFrame(ndvi_df, 
                                geometry=gpd.points_from_xy(ndvi_df.x, ndvi_df.y), 
                                crs="EPSG:2263")
    output_path = '../output/sentinel2.parquet'
    gdf_ndvi.to_parquet(output_path)
    print(f"Sentinel-2 NDVI data saved to: {output_path}")
    return gdf_ndvi

# ----------------------------------------------------------------------
# 04 Processing LandSat Data (LandSat and Albedo)
# ----------------------------------------------------------------------

def process_landsat_data():
    print("Processing Landsat data for LST and Albedo...")
    warnings.filterwarnings('ignore')
    lower_left = (40.75, -74.01)
    upper_right = (40.88, -73.86)
    bounds = (lower_left[1], lower_left[0], upper_right[1], upper_right[0])
    time_window = "2021-06-01/2021-09-01"
    
    stac = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    search = stac.search(
        bbox=bounds,
        datetime=time_window,
        collections=["landsat-c2-l2"],
        query={"eo:cloud_cover": {"lt": 50}, "platform": {"in": ["landsat-8"]}},
    )
    items = list(search.get_items())
    print("Number of Landsat scenes found:", len(items))
    
    resolution = 30
    # Load the optical bands needed for albedo: 
    # blue (B2), red (B4), nir08 (B5), swir16 (B6), swir22 (B7)
    data1 = stac_load(
        items,
        bands=["blue", "red", "nir08", "swir16", "swir22"],
        crs="EPSG:2263",
        resolution=resolution,
        chunks={"x": 2048, "y": 2048},
        dtype="uint16",
        patch_url=planetary_computer.sign,
        bbox=bounds
    )
    # Load the thermal band for LST calculation
    data2 = stac_load(
        items,
        bands=["lwir11"],
        crs="EPSG:2263",
        resolution=resolution,
        chunks={"x": 2048, "y": 2048},
        dtype="uint16",
        patch_url=planetary_computer.sign,
        bbox=bounds
    )
    
    # Scale optical bands to surface reflectance
    scale1 = 0.0000275
    offset1 = -0.2
    data1 = data1.astype(float) * scale1 + offset1
    
    # Calculate broadband albedo using the formula:
    # alpha = 0.356*blue + 0.130*red + 0.373*nir08 + 0.085*swir16 + 0.072*swir22 - 0.018
    albedo = (0.356 * data1["blue"] +
              0.130 * data1["red"] +
              0.373 * data1["nir08"] +
              0.085 * data1["swir16"] +
              0.072 * data1["swir22"] - 0.018)
    
    # Process thermal band for LST calculation
    scale2 = 0.00341802
    offset2 = 149.0
    kelvin_celsius = 273.15
    data2 = data2.astype(float) * scale2 + offset2 - kelvin_celsius
    
    # Select the scene closest to the target date for both LST and albedo
    target_date = np.datetime64("2021-07-24")
    time_diffs = abs(data2.time - target_date)
    closest_time_index = int(time_diffs.argmin())
    
    lst_slice = data2.isel(time=closest_time_index)
    albedo_slice = albedo.isel(time=closest_time_index)
    
    # Convert LST slice to DataFrame and GeoDataFrame
    lst_df = lst_slice.to_dataframe().reset_index()
    lst_df = lst_df.rename(columns={'lwir11': 'LST'})
    gdf_lst = gpd.GeoDataFrame(lst_df, 
                               geometry=gpd.points_from_xy(lst_df.x, lst_df.y), 
                               crs="EPSG:2263")
    gdf_lst = gdf_lst[['LST', 'geometry']]
    
    # Convert albedo slice to DataFrame and GeoDataFrame
    if albedo_slice.name is None:
        albedo_slice = albedo_slice.rename("Albedo")
    albedo_df = albedo_slice.to_dataframe().reset_index()
    # If the computed DataArray does not have a name, assign 'Albedo'
    if albedo_slice.name is None:
        albedo_df = albedo_df.rename(columns={0: 'Albedo'})
    else:
        albedo_df = albedo_df.rename(columns={albedo_slice.name: 'Albedo'})
    gdf_albedo = gpd.GeoDataFrame(albedo_df, 
                                  geometry=gpd.points_from_xy(albedo_df.x, albedo_df.y), 
                                  crs="EPSG:2263")
    gdf_albedo = gdf_albedo[['Albedo', 'geometry']]
    
    # Save the LST and albedo GeoDataFrames as parquet files
    lst_output_path = '../output/landsat_lst.parquet'
    albedo_output_path = '../output/landsat_albedo.parquet'
    gdf_lst.to_parquet(lst_output_path)
    gdf_albedo.to_parquet(albedo_output_path)
    print(f"Landsat LST data saved to: {lst_output_path}")
    print(f"Landsat albedo data saved to: {albedo_output_path}")
    
    # Optional: Print sample data from the saved parquet files
    lst_sample = pq.ParquetFile(lst_output_path).read_row_group(0, columns=['LST']).to_pandas()
    albedo_sample = pq.ParquetFile(albedo_output_path).read_row_group(0, columns=['Albedo']).to_pandas()
    print("Sample from Landsat LST data:")
    print(lst_sample.head())
    print("Sample from Landsat albedo data:")
    print(albedo_sample.head())
    
    return gdf_lst, gdf_albedo
# ----------------------------------------------------------------------
# 05 Processing Weather Data
# ----------------------------------------------------------------------
def process_weather_data():
    print("Processing weather data...")
    df_bronx = pd.read_excel("../input/NY_Mesonet_Weather.xlsx", sheet_name="Bronx")
    df_manhattan = pd.read_excel("../input/NY_Mesonet_Weather.xlsx", sheet_name="Manhattan")
    df_bronx["location"] = "Bronx"
    df_manhattan["location"] = "Manhattan"
    df_weather = pd.concat([df_bronx, df_manhattan], ignore_index=True)
    print("Sample weather data:")
    print(df_weather.head())
    df_weather['Date / Time'] = pd.to_datetime(df_weather['Date / Time'], errors='coerce')
    output_path = "../output/weather_data_processed.parquet"
    df_weather.to_parquet(output_path, index=False)
    print(f"Weather data saved to: {output_path}")
    return df_weather

# ----------------------------------------------------------------------
# 06 Feature Engineering
# ----------------------------------------------------------------------
def feature_engineering():
    print("Performing feature engineering...")
    df_uhi = gpd.read_parquet("../output/uhi_data_processed.parquet")
    gdf_buildings = gpd.read_parquet("../output/building_data_processed.parquet")
    gdf_ndvi = gpd.read_parquet("../output/sentinel2.parquet")
    gdf_lst = gpd.read_parquet("../output/landsat_lst.parquet")
    df_weather = pd.read_parquet("../output/weather_data_processed.parquet")
    
    # Create a 100-meter buffer around each UHI point
    df_uhi['buffer'] = df_uhi.geometry.buffer(100)
    
    # Spatial join with building footprints and aggregate building features
    joined = gpd.sjoin(gdf_buildings, df_uhi.set_geometry('buffer'), how='inner', predicate='intersects')
    agg_buildings = joined.groupby('index_right').agg(
        building_count=('heightroof', 'count'),
        mean_height=('heightroof', 'mean'),
        total_area=('calculated_area_sqm', 'sum')
    ).reset_index()
    df_uhi = df_uhi.reset_index().merge(agg_buildings, left_index=True, right_on='index_right', how='left')
    
    buffer_area = 31416  # approximate area for 100m radius buffer in m²
    df_uhi['building_area_ratio'] = df_uhi['total_area'] / buffer_area
    
    # Spatial join with NDVI pixels and aggregate NDVI features
    joined_ndvi = gpd.sjoin(gdf_ndvi, df_uhi.set_geometry('buffer'), how='inner', predicate='intersects')
    agg_ndvi = joined_ndvi.groupby('index_right').agg(
        mean_ndvi=('NDVI', 'mean'),
        median_ndvi=('NDVI', 'median'),
        std_ndvi=('NDVI', 'std')
    ).reset_index()
    df_uhi = df_uhi.merge(agg_ndvi, left_on='index', right_on='index_right', how='left')
    
    print("Feature engineering completed. Sample features:")
    print(df_uhi.head())
    return df_uhi, df_weather

# ----------------------------------------------------------------------
# Additional Preprocessing for Modelling
# ----------------------------------------------------------------------
def prepare_features(df_model):
    """
    Impute missing building features with 0, impute NDVI columns with their citywide mean,
    clip NDVI values to [-1, 1], and compute log-transformed features.
    """
    df_model['building_count'] = df_model['building_count'].fillna(0)
    df_model['mean_height'] = df_model['mean_height'].fillna(0)
    df_model['total_area'] = df_model['total_area'].fillna(0)
    df_model['building_area_ratio'] = df_model['building_area_ratio'].fillna(0)
    
    ndvi_cols = ['mean_ndvi', 'median_ndvi', 'std_ndvi']
    for col in ndvi_cols:
        df_model[col] = df_model[col].fillna(df_model[col].mean())
    
    df_model['mean_ndvi'] = df_model['mean_ndvi'].clip(-1, 1)
    df_model['median_ndvi'] = df_model['median_ndvi'].clip(-1, 1)
    
    df_model['log_total_area'] = np.log1p(df_model['total_area'])
    df_model['log_building_area_ratio'] = np.log1p(df_model['building_area_ratio'])
    return df_model

# ----------------------------------------------------------------------
# 07 Model Training and Evaluation using LightGBM
# ----------------------------------------------------------------------
def run_modelling():
    print("Running modelling...")
    df_uhi, df_weather = feature_engineering()
    # For modelling, we use the engineered UHI dataframe as our model dataset.
    df_model = df_uhi.copy()
    df_model = prepare_features(df_model)
    
    # Define predictor features (exclude non-predictors)
    features = [col for col in df_model.columns if col not in ['UHI Index', 'datetime', 'Longitude', 'Latitude']]
    X = df_model[features]
    y = df_model['UHI Index']
    
    # Split data into training and testing sets (70/30 split)
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
    best_params = grid.best_params_
    best_cv_score = grid.best_score_
    
    y_pred = best_model.predict(X_test)
    test_r2 = r2_score(y_test, y_pred)
    
    print("\nBest Parameters:", best_params)
    print("Best Cross-Validation R²:", best_cv_score)
    print("Test R²:", test_r2)
    
    return best_model

# ----------------------------------------------------------------------
# Optional Exports (Landsat LST / Sentinel-2 GeoTIFF)
# ----------------------------------------------------------------------
def export_landsat_lst():
    print("Exporting Landsat LST as needed...")
    # Additional export steps can be added here if required.
    pass

def export_sentinel2_geotiff():
    print("Exporting Sentinel-2 NDVI as GeoTIFF...")
    gdf_ndvi = gpd.read_parquet("../output/sentinel2.parquet")
    # Placeholder: Use rasterio to export GeoTIFF from NDVI data if needed.
    pass

# ----------------------------------------------------------------------
# Main Workflow
# ----------------------------------------------------------------------
def main():
    # Data processing steps
    uhi_data = read_target_variables()
    building_data = process_building_footprints()
    ndvi_data = process_sentinel2_data()
    landsat_data = process_landsat_data()
    weather_data = process_weather_data()
    
    # Feature engineering and modelling
    best_model = run_modelling()
    
    # Optional exports
    # export_landsat_lst()
    # export_sentinel2_geotiff()
    
    print("All processing steps completed successfully.")
    return best_model

if __name__ == "__main__":
    main()
