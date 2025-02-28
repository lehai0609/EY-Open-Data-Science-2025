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

'''
Is the single script you run to orchestrate everything in sequence.
'''
#!/usr/bin/env python3
"""
main_pipeline.py

Orchestrates the full data science pipeline for the EY Open Data Science 2025 Project.

Modes:
  - train: Trains and saves the model
  - predict: Applies the model to validation data

Usage:
  python main_pipeline.py [train|predict]
"""

import os
import pickle
import sys

def main(mode='train'):
    """
    Orchestrates the full data science pipeline.
    
    Parameters:
        mode: 'train' for training, 'predict' for prediction
    """
    if mode == 'train':
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
        
        # --- Data Ingestion ---
        print("Starting Data Ingestion Stage...")
        gdf_uhi = read_target_variables()
        gdf_buildings = read_building_footprints()
        df_energy_star = read_energy_star_data()
        df_weather = read_weather_data()
        gdf_svi = read_social_vulnerability_index()

        # Merge ENERGY STAR scores into building footprints
        print("Merging ENERGY STAR scores into building footprints...")
        gdf_buildings = gdf_buildings.merge(df_energy_star, on='bin', how='left')
        
        # --- Data Processing (Remote Sensing) ---
        print("Starting Data Processing Stage for Remote Sensing Data...")
        gdf_ndvi = compute_ndvi_sentinel2()
        gdf_lst, gdf_albedo = compute_landsat_lst_albedo()
        
        # --- Feature Engineering ---
        print("Starting Feature Engineering Stage...")
        df_engineered, df_weather_processed = feature_engineering(
            gdf_uhi, gdf_buildings, gdf_ndvi, gdf_albedo, df_weather, gdf_svi, gdf_lst
        )
        
        # --- Modeling ---
        print("Starting Modeling Stage...")
        best_model = run_modelling(df_engineered)
        
        # Save the model
        os.makedirs('output', exist_ok=True)
        model_path = 'output/best_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(best_model, f)
        
        print("Pipeline completed successfully.")
        print(f"Best model saved to {model_path}")
        
        
    elif mode == 'predict':
        # Import prediction function
        from core.prediction import run_prediction_pipeline
        
        validation_file_path = 'output/Submission_template_UHI2025-v2.csv'
        model_path = 'output/best_model.pkl'
        output_path = 'output/UHI_predictions.csv'
        
        # Check if files exist
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}")
            print("Please run the training pipeline first.")
            return
        
        if not os.path.exists(validation_file_path):
            print(f"Error: Validation file not found at {validation_file_path}")
            return
        
        # Run prediction pipeline
        run_prediction_pipeline(validation_file_path, model_path, output_path)
        
    else:
        print(f"Invalid mode: {mode}. Use 'train' or 'predict'.")

if __name__ == "__main__":
    mode = 'train'
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    
    main(mode)
