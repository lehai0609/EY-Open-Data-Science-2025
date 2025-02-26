'''
Is the single script you run to orchestrate everything in sequence.
'''
#!/usr/bin/env python3
"""
main_pipeline.py

Orchestrates the full data science pipeline for the EY Open Data Science 2025 Project.

Modules:
  - data_ingestion: Reads raw data (target variables, building footprints, ENERGY STAR, weather)
  - data_processing: Processes remote sensing data (computes Sentinel-2 NDVI, Landsat LST & Albedo)
  - feature_engineering: Aggregates and merges features from all data layers
  - modeling: Trains and evaluates a LightGBM model on the engineered features
"""

# Import functions from the core modules
from core.data_ingestion import (
    read_target_variables,
    read_building_footprints,
    read_energy_star_data,
    read_weather_data,
    read_social_vulnerability_index
)
from core.data_processing import (
    compute_ndvi_sentinel2,
    compute_landsat_lst_albedo
)
from core.feature_engineering import feature_engineering
from core.modeling import run_modelling

def main():
    # --- Data Ingestion ---
    print("Starting Data Ingestion Stage...")
    gdf_uhi = read_target_variables()           # UHI target variables
    gdf_buildings = read_building_footprints()    # Building footprints
    df_energy_star = read_energy_star_data()      # ENERGY STAR scores
    df_weather = read_weather_data()              # Weather data
    gdf_svi = read_social_vulnerability_index()   # Social Vulnerability Index data

    # Merge ENERGY STAR scores into building footprints (assumes common key 'bin')
    print("Merging ENERGY STAR scores into building footprints...")
    gdf_buildings = gdf_buildings.merge(df_energy_star, on='bin', how='left')
    
    # --- Data Processing (Remote Sensing) ---
    print("Starting Data Processing Stage for Remote Sensing Data...")
    gdf_ndvi = compute_ndvi_sentinel2()           # Sentinel-2 NDVI computation
    gdf_lst, gdf_albedo = compute_landsat_lst_albedo()  # Landsat LST and Albedo computation
    
    # --- Feature Engineering ---
    print("Starting Feature Engineering Stage...")
    # Pass df_person along with other datasets for integrated feature engineering.
    df_engineered, df_weather_processed = feature_engineering(
        gdf_uhi, gdf_buildings, gdf_ndvi, gdf_albedo, df_weather, gdf_svi, gdf_lst
    )
    
    # --- Modeling ---
    print("Starting Modeling Stage...")
    best_model = run_modelling(df_engineered)
    
    print("Pipeline completed successfully.")
    print("Best model:", best_model)

if __name__ == "__main__":
    main()
