'''
Contains functions that transform or process that data
'''
#!/usr/bin/env python3
"""
data_processing.py

This module performs the heavy-lifting of processing raw remote sensing data.
It includes:
  - compute_ndvi_sentinel2: Querying Sentinel-2 data via STAC and computing NDVI.
  - compute_landsat_lst_albedo: Querying Landsat data via STAC and computing LST and broadband albedo.

Parameters (for both functions) include:
  - bbox: Bounding box (min_lon, min_lat, max_lon, max_lat)
  - time_window: ISO time range (e.g., "YYYY-MM-DD/YYYY-MM-DD")
  - target_date: Date string to select the scene closest to the target date
  - resolution: Spatial resolution in meters
"""

import geopandas as gpd
import numpy as np
import pandas as pd
import warnings
import xarray as xr
import pystac_client
import planetary_computer
from odc.stac import stac_load

def compute_ndvi_sentinel2(bbox=(-74.01, 40.75, -73.86, 40.88),
                           time_window="2021-07-01/2021-08-01",
                           target_date="2021-07-24",
                           resolution=10):
    """
    Queries Sentinel-2 data via STAC, computes NDVI and additional vegetation metrics,
    and returns a GeoDataFrame.
    
    Parameters:
      bbox: tuple, bounding box in the order (min_lon, min_lat, max_lon, max_lat)
      time_window: str, time range (e.g., "YYYY-MM-DD/YYYY-MM-DD")
      target_date: str, target date to select the closest scene (e.g., "YYYY-MM-DD")
      resolution: int, spatial resolution in meters
      
    Returns:
      GeoDataFrame with NDVI values, enhanced vegetation metrics, and associated geometry.
    """
    print("Computing NDVI and vegetation metrics from Sentinel-2 data...")
    warnings.filterwarnings('ignore')
    
    # Open STAC client and search for Sentinel-2 data
    stac = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    search = stac.search(
        bbox=bbox,
        datetime=time_window,
        collections=["sentinel-2-l2a"],
        query={"eo:cloud_cover": {"lt": 30}},
    )
    items = list(search.get_items())
    print("Number of Sentinel-2 scenes found:", len(items))
    
    # Load data using stac_load function
    data = stac_load(
        items,
        bands=["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12", "SCL"],
        crs="EPSG:2263",
        resolution=resolution,
        chunks={"x": 2048, "y": 2048},
        dtype="uint16",
        patch_url=planetary_computer.sign,
        bbox=bbox
    )
    
    # Compute NDVI using bands B08 (NIR) and B04 (red)
    valid_mask = (data.SCL == 4) | (data.SCL == 5)  # Valid pixels (vegetation, bare soil)
    ndvi = (data.B08 - data.B04) / (data.B08 + data.B04 + 1e-6)
    ndvi = ndvi.where(valid_mask, other=np.nan)
    
    # NEW: Compute Enhanced Vegetation Index (EVI)
    # EVI = G * ((NIR - RED) / (NIR + C1 * RED - C2 * BLUE + L))
    # Where G=2.5, C1=6, C2=7.5, L=1
    G = 2.5
    C1 = 6.0
    C2 = 7.5
    L = 1.0
    evi = G * ((data.B08 - data.B04) / (data.B08 + C1 * data.B04 - C2 * data.B02 + L + 1e-6))
    evi = evi.where(valid_mask, other=np.nan)
    
    # NEW: Compute NDWI (Normalized Difference Water Index) using Green and NIR bands
    ndwi = (data.B03 - data.B08) / (data.B03 + data.B08 + 1e-6)
    ndwi = ndwi.where(valid_mask, other=np.nan)
    
    # Compute all indices
    ndvi = ndvi.compute()
    evi = evi.compute()
    ndwi = ndwi.compute()
    
    # Select scene closest to the target_date
    target_dt = np.datetime64(target_date)
    time_diffs = abs(data.time - target_dt)
    closest_time_index = int(time_diffs.argmin())
    
    ndvi_slice = ndvi.isel(time=closest_time_index)
    evi_slice = evi.isel(time=closest_time_index)
    ndwi_slice = ndwi.isel(time=closest_time_index)
    
    # Convert to DataFrame
    ndvi_df = ndvi_slice.to_dataframe(name='NDVI').reset_index()
    ndvi_df['EVI'] = evi_slice.to_dataframe(name='EVI').reset_index()['EVI']
    ndvi_df['NDWI'] = ndwi_slice.to_dataframe(name='NDWI').reset_index()['NDWI']
    
    # NEW: Create vegetation class categories
    ndvi_df['veg_class'] = pd.cut(ndvi_df['NDVI'], 
                                 bins=[-1, 0.2, 0.4, 0.6, 1], 
                                 labels=['No Vegetation', 'Low Vegetation', 'Moderate Vegetation', 'High Vegetation'])
    
    # Create GeoDataFrame
    gdf_ndvi = gpd.GeoDataFrame(
        ndvi_df,
        geometry=gpd.points_from_xy(ndvi_df.x, ndvi_df.y),
        crs="EPSG:2263"
    )
    
    print("NDVI and vegetation metrics computation complete.")
    return gdf_ndvi

def compute_landsat_lst_albedo(bbox=(-74.01, 40.75, -73.86, 40.88),
                               time_window="2021-06-01/2021-09-01",
                               target_date="2021-07-24",
                               resolution=30):
    """
    Queries Landsat data via STAC, computes Land Surface Temperature (LST) and broadband albedo,
    and returns two GeoDataFrames.
    
    Parameters:
      bbox: tuple, bounding box in the order (min_lon, min_lat, max_lon, max_lat)
      time_window: str, time range (e.g., "YYYY-MM-DD/YYYY-MM-DD")
      target_date: str, target date to select the closest scene (e.g., "YYYY-MM-DD")
      resolution: int, spatial resolution in meters
      
    Returns:
      Tuple of GeoDataFrames: (gdf_lst, gdf_albedo)
    """
    print("Computing LST and Albedo from Landsat data...")
    warnings.filterwarnings('ignore')
    
    stac = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    search = stac.search(
        bbox=bbox,
        datetime=time_window,
        collections=["landsat-c2-l2"],
        query={"eo:cloud_cover": {"lt": 50}, "platform": {"in": ["landsat-8"]}},
    )
    items = list(search.get_items())
    print("Number of Landsat scenes found:", len(items))
    
    # Load optical bands for albedo computation
    data1 = stac_load(
        items,
        bands=["blue", "red", "nir08", "swir16", "swir22"],
        crs="EPSG:2263",
        resolution=resolution,
        chunks={"x": 2048, "y": 2048},
        dtype="uint16",
        patch_url=planetary_computer.sign,
        bbox=bbox
    )
    # Load thermal band for LST computation
    data2 = stac_load(
        items,
        bands=["lwir11"],
        crs="EPSG:2263",
        resolution=resolution,
        chunks={"x": 2048, "y": 2048},
        dtype="uint16",
        patch_url=planetary_computer.sign,
        bbox=bbox
    )
    
    # Process optical bands: scale and compute broadband albedo
    scale1 = 0.0000275
    offset1 = -0.2
    data1 = data1.astype(float) * scale1 + offset1
    albedo = (0.356 * data1["blue"] +
              0.130 * data1["red"] +
              0.373 * data1["nir08"] +
              0.085 * data1["swir16"] +
              0.072 * data1["swir22"] - 0.018)
    
    # Process thermal band: scale and compute LST
    scale2 = 0.00341802
    offset2 = 149.0
    kelvin_celsius = 273.15
    data2 = data2.astype(float) * scale2 + offset2 - kelvin_celsius
    
    # Select scene closest to the target_date
    target_dt = np.datetime64(target_date)
    time_diffs = abs(data2.time - target_dt)
    closest_time_index = int(time_diffs.argmin())
    lst_slice = data2.isel(time=closest_time_index)
    albedo_slice = albedo.isel(time=closest_time_index)
    
    # Convert LST slice to a GeoDataFrame
    lst_df = lst_slice.to_dataframe().reset_index().rename(columns={'lwir11': 'LST'})
    gdf_lst = gpd.GeoDataFrame(
        lst_df,
        geometry=gpd.points_from_xy(lst_df.x, lst_df.y),
        crs="EPSG:2263"
    )
    gdf_lst = gdf_lst[['LST', 'geometry']]
    
    # Convert albedo slice to a GeoDataFrame
    if albedo_slice.name is None:
        albedo_slice = albedo_slice.rename("Albedo")
    albedo_df = albedo_slice.to_dataframe().reset_index()
    if albedo_slice.name is None:
        albedo_df = albedo_df.rename(columns={0: 'Albedo'})
    else:
        albedo_df = albedo_df.rename(columns={albedo_slice.name: 'Albedo'})
    gdf_albedo = gpd.GeoDataFrame(
        albedo_df,
        geometry=gpd.points_from_xy(albedo_df.x, albedo_df.y),
        crs="EPSG:2263"
    )
    gdf_albedo = gdf_albedo[['Albedo', 'geometry']]
    
    print("LST and Albedo computation complete.")
    return gdf_lst, gdf_albedo

# Optional testing when running this module directly
if __name__ == "__main__":
    ndvi_gdf = compute_ndvi_sentinel2()
    print("Sentinel-2 NDVI sample:")
    print(ndvi_gdf.head())
    
    lst_gdf, albedo_gdf = compute_landsat_lst_albedo()
    print("Landsat LST sample:")
    print(lst_gdf.head())
    print("Landsat Albedo sample:")
    print(albedo_gdf.head())
