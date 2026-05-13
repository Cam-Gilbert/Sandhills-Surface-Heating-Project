#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 16:10:51 2026

@author: cg154122
"""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
import rioxarray
from rasterio.crs import CRS
from rasterio.enums import Resampling
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd


def resample_ndvi(ndvi, lst):
    '''
    NDVI is at a higher resolution than LST data. So we need to resample it to match the 1km LST data
    
    @param ndvi xarray array of NDVI at 500m res
    @param lst xarray array of LST at 1km res
    @return NDVI resampled to 1km
    '''
    return ndvi.rio.reproject_match(lst, resampling = Resampling.average)


OUTPUT_CRS = CRS.from_epsg(32617)

lst_datasets = []
ndvi_datasets = []

files = sorted(Path('data/processed/').rglob("*.tif"))

for f in files:
    if "lst_day" in str(f):
        da = xr.open_dataarray(f, engine="rasterio").rio.write_crs(OUTPUT_CRS)
        lst_datasets.append(da)
    elif "ndvi_jja" in str(f):
        ndvi = xr.open_dataarray(f, engine="rasterio").rio.write_crs(OUTPUT_CRS)
        ndvi_datasets.append(ndvi)


lst_combined = xr.concat(lst_datasets, dim='new_dim')
ndvi_combined = xr.concat(ndvi_datasets, dim='new_dim')

lst_average = lst_combined.mean(dim='new_dim', skipna=True)
ndvi_average = ndvi_combined.mean(dim='new_dim', skipna=True)

# resample ndvi
ndvi_resample = resample_ndvi(ndvi_average, lst_average)

eco = gpd.read_file("data/sandhills_shapefile/us_eco_l4_no_st.shp")
sandhills = eco[eco["US_L4CODE"] == "65c"]
sandhills = sandhills.to_crs(OUTPUT_CRS)

buffered = sandhills.buffer(75000) # meter
control = gpd.GeoDataFrame(geometry=buffered, crs=sandhills.crs)


# Get only values within sandhills region
lst_clipped_sandhills = lst_average.rio.clip(
    sandhills.geometry,
    crs=sandhills.crs,
    drop=True,
    all_touched=False
)

ndvi_clipped_sandhills = ndvi_resample.rio.clip(
    sandhills.geometry,
    crs=sandhills.crs,
    drop=True,
    all_touched=False
)

# Get valus in control region
lst_clipped_control = lst_average.rio.clip(
    control.geometry,
    crs=control.crs,
    drop=True,
    all_touched=False
)

ndvi_clipped_control = ndvi_resample.rio.clip(
    control.geometry,
    crs=control.crs,
    drop=True,
    all_touched=False
)

# flatten and drop nans
lst_flat_sandhills = lst_clipped_sandhills.values.ravel()
ndvi_flat_sandhills = ndvi_clipped_sandhills.values.ravel()

lst_flat_control = lst_clipped_control.values.ravel()
ndvi_flat_control = ndvi_clipped_control.values.ravel()

valid = ~np.isnan(lst_flat_sandhills) & ~np.isnan(ndvi_flat_sandhills)
lst_sandhills = lst_flat_sandhills[valid]
ndvi_sandhills = ndvi_flat_sandhills[valid]

valid_control = ~np.isnan(lst_flat_control) & ~np.isnan(ndvi_flat_control)
lst_control = lst_flat_control[valid_control]
ndvi_control = ndvi_flat_control[valid_control]

# PLOT
def plot_scatter(ndvi, lst, study_region = "", save = False):
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.scatter(ndvi, lst,
               s=1, 
               alpha=0.3,
               color="forestgreen",
               rasterized=True)
    
    # plot line of best fit
    # m, b = np.polyfit(ndvi, lst, 1)
    # x_line = np.linspace(ndvi.min(), ndvi.max(), 100)
    # ax.plot(x_line, m * x_line + b, color="black", linewidth=1.5,
    #         label=f"slope = {m:.1f} °C / NDVI unit")
    
    ax.set_xlabel("Mean JJA NDVI", fontsize=15)
    ax.set_ylabel("Mean JJA Daytime LST (°C)", fontsize=15)
    ax.set_title(f"LST vs NDVI — {study_region} Region", fontsize=20)
    ax.legend(fontsize = 13)
    
    # correlation
    # r = np.corrcoef(ndvi, lst)[0, 1]
    # ax.text(0.10, 0.95, f"r = {r:.3f}", transform=ax.transAxes,
    #         fontsize=13, verticalalignment="top")
    
    plt.tight_layout()
    if save:
        plt.savefig(f"figures/lst_ndvi_scatter_{study_region}_nofit.png")
    plt.show()


plot_scatter(ndvi_control, lst_control, study_region = "Control", save = True)