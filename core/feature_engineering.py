'''
Contains functions that load data from ../input folder
'''
#!/usr/bin/env python3
"""
data_ingestion.py

This module is responsible for reading the raw input data:
  - Target variable data (UHI)
  - Building footprints data
  - ENERGY STAR score data
  - New York weather data

Note:
  Sentinel-2 and Landsat data heavy-lifting (e.g. STAC queries and remote sensing calculations)
  will be handled in data_processing.py.
"""

import os
import pandas as pd
import geopandas as gpd

# ----------------------------------------------------------------------
# 01 Read Target Variables
# ----------------------------------------------------------------------
def read_target_variables(filepath="input/Training_data_uhi_index_2025-02-18.csv"):
    """
    Reads the UHI target variables from a CSV file,
    converts the datetime column, creates a GeoDataFrame, and reprojects it to EPSG:2263.
    """
    print("Reading UHI target variables from:", filepath)
    df = pd.read_csv(filepath)
    df['datetime'] = pd.to_datetime(df['datetime'], format="%d-%m-%Y %H:%M")
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.Longitude, df.Latitude),
        crs="EPSG:4326"
    )
    gdf = gdf.to_crs("EPSG:2263")
    return gdf

# ----------------------------------------------------------------------
# 02 Read Building Footprints
# ----------------------------------------------------------------------
def read_building_footprints(filepath="input/Building Footprints_20250131.geojson"):
    """
    Reads building footprints from a GeoJSON file, converts key columns to numeric,
    applies minimal cleaning, and reprojects to EPSG:2263.
    """
    print("Reading building footprints from:", filepath)
    gdf = gpd.read_file(filepath)
    
    # Convert selected columns to numeric
    cols_to_convert = ['shape_area', 'heightroof', 'cnstrct_yr', 'groundelev']
    for col in cols_to_convert:
        gdf[col] = pd.to_numeric(gdf[col], errors='coerce')
    gdf['lstmoddate'] = pd.to_datetime(gdf['lstmoddate'], errors='coerce')
    
    # Reproject and compute additional area fields if needed
    gdf = gdf.to_crs(epsg=2263)
    gdf['calculated_area_sqft'] = gdf.geometry.area
    gdf['calculated_area_sqm'] = gdf['calculated_area_sqft'] * 0.092903
    
    # Drop columns that are not needed for downstream processing
    columns_to_remove = ['name', 'base_bbl', 'mpluto_bbl', 'cnstrct_yr', 'doitt_id',
                         'geomsource', 'lststatype', 'shape_len', 'globalid',
                         'feat_code', 'lstmoddate', 'calculated_area_sqft']
    gdf.drop(columns=columns_to_remove, inplace=True, errors='ignore')
    
    # Fill missing values in key numeric fields
    gdf['heightroof'] = gdf['heightroof'].fillna(gdf['heightroof'].median())
    gdf['groundelev'] = gdf['groundelev'].fillna(gdf['groundelev'].median())
    return gdf

# ----------------------------------------------------------------------
# 03 Read ENERGY STAR Score Data
# ----------------------------------------------------------------------
def read_energy_star_data(filepath="input/Energy_and_Water_Data_2021.csv"):
    """
    Reads ENERGY STAR score data from a CSV file and prepares it for merging.
    """
    print("Reading ENERGY STAR data from:", filepath)
    df = pd.read_csv(filepath)
    df = df[['NYC Building Identification Number (BIN)', 'ENERGY STAR Score']]
    df = df.rename(columns={'NYC Building Identification Number (BIN)': 'bin'})
    df['ENERGY STAR Score'] = pd.to_numeric(df['ENERGY STAR Score'], errors='coerce')
    return df

# ----------------------------------------------------------------------
# 04 Read New York Weather Data
# ----------------------------------------------------------------------
def read_weather_data(filepath="input/NY_Mesonet_Weather.xlsx"):
    """
    Reads New York weather data from an Excel file (from both Bronx and Manhattan sheets),
    concatenates them, cleans column names, and returns a DataFrame.
    """
    print("Reading New York weather data from:", filepath)
    df_bronx = pd.read_excel(filepath, sheet_name="Bronx")
    df_manhattan = pd.read_excel(filepath, sheet_name="Manhattan")
    df_bronx["location"] = "Bronx"
    df_manhattan["location"] = "Manhattan"
    
    df_weather = pd.concat([df_bronx, df_manhattan], ignore_index=True)
    df_weather['Date / Time'] = pd.to_datetime(df_weather['Date / Time'], errors='coerce')
    
    # Clean column names
    df_weather = pd.get_dummies(df_weather, columns=['location'], prefix='loc', dtype=int)
    df_weather.columns = df_weather.columns.str.replace(r'[\[\]]', '', regex=True)
    df_weather.columns = df_weather.columns.str.replace(' ', '_')
    df_weather.columns = df_weather.columns.str.replace(r'[^\w]', '', regex=True)
    return df_weather

# ----------------------------------------------------------------------
# 05 Read socio vulnerability index data
# ----------------------------------------------------------------------
def read_social_vulnerability_index(filepath="input/SVI2022_NEWYORK_tract.gdb"):
    """
    Reads the Social Vulnerability Index (SVI) data from a geodatabase file.
    The CDC/ATSDR SVI identifies communities that may need support before, during, or after
    disasters based on 16 social factors, including socioeconomic status, household composition,
    disability, language, etc.
    
    Returns:
        gdf_svi: GeoDataFrame with SVI data for New York County
    """
    print("Reading Social Vulnerability Index (SVI) data from:", filepath)
    
    # Read the GDB file - the layer name might need adjustment based on the actual file
    gdf_svi = gpd.read_file(filepath)
    
    # Keep only the useful columns for our analysis - these might need adjustment
    # based on the actual SVI data structure
    useful_columns = [
        'GEOID', 'geometry',  # Identification and geometry
        'RPL_THEME1', 'RPL_THEME2', 'RPL_THEME3', 'RPL_THEME4', 'RPL_THEMES',  # Overall themes
        'RPL_POVTY', 'RPL_UNEMP', 'RPL_PCI', 'RPL_NOHSDP',  # Socioeconomic theme
        'RPL_AGE65', 'RPL_AGE17', 'RPL_DISABL', 'RPL_SNGPNT',  # Household composition theme
        'RPL_MINRTY', 'RPL_LIMENG',  # Minority status/Language theme
        'RPL_MUNIT', 'RPL_MOBILE', 'RPL_CROWD', 'RPL_NOVEH', 'RPL_GROUPQ'  # Housing/Transportation theme
    ]
    
    # Filter columns if they exist in the dataset
    available_columns = [col for col in useful_columns if col in gdf_svi.columns]
    if set(available_columns) != set(useful_columns):
        missing = set(useful_columns) - set(available_columns)
        print(f"Warning: Some expected SVI columns not found: {missing}")
    
    gdf_svi = gdf_svi[available_columns]
    
    # Ensure proper projection to EPSG:2263 to match other data
    if gdf_svi.crs is not None and gdf_svi.crs != "EPSG:2263":
        gdf_svi = gdf_svi.to_crs("EPSG:2263")
    elif gdf_svi.crs is None:
        print("Warning: SVI data has no CRS information. Assuming EPSG:4326 (WGS84).")
        gdf_svi.crs = "EPSG:4326"
        gdf_svi = gdf_svi.to_crs("EPSG:2263")
    
    return gdf_svi

# ----------------------------------------------------------------------
# Optional: Test the data ingestion functions when running directly.
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("Testing data ingestion functions...")
    uhi_gdf = read_target_variables()
    buildings_gdf = read_building_footprints()
    energy_df = read_energy_star_data()
    weather_df = read_weather_data()
    
    print("UHI sample:")
    print(uhi_gdf.head())
    print("\nBuilding footprints sample:")
    print(buildings_gdf.head())
    print("\nENERGY STAR sample:")
    print(energy_df.head())
    print("\nWeather data sample:")
    print(weather_df.head())
    
    print("Data ingestion completed successfully.")
