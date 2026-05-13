#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 14:35:08 2026

@author: cg154122
"""
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import xarray as xr
from pathlib import Path
from rasterio.crs import CRS
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd

#da = xr.open_dataarray("data/processed/lst_day_2020_06.tif", engine="rasterio")
OUTPUT_CRS = CRS.from_epsg(32617)

day_datasets = []
night_datasets = []
dtr_datasets = []
evi_datasets = []
ndvi_datasets = []
daily_files = []

files = sorted(Path('data/processed/').rglob("*.tif"))

for f in files:
    if "lst_day" in str(f):
        da = xr.open_dataarray(f, engine="rasterio").rio.write_crs(OUTPUT_CRS)
        day_datasets.append(da)
    elif "lst_night" in str(f):
        da = xr.open_dataarray(f, engine="rasterio").rio.write_crs(OUTPUT_CRS)
        night_datasets.append(da)
    elif "lst_dtr" in str(f):
        da = xr.open_dataarray(f, engine="rasterio").rio.write_crs(OUTPUT_CRS)
        dtr_datasets.append(da)
    elif "evi_jja" in str(f):
        evi = xr.open_dataarray(f, engine="rasterio").rio.write_crs(OUTPUT_CRS)
        evi_datasets.append(evi)
    elif "ndvi_jja" in str(f):
        ndvi = xr.open_dataarray(f, engine="rasterio").rio.write_crs(OUTPUT_CRS)
        ndvi_datasets.append(ndvi)
    else:
        daily_files.append(f)
           
    
day_combined = xr.concat(day_datasets, dim='new_dim')
night_combined = xr.concat(night_datasets, dim='new_dim')
dtr_combined = xr.concat(dtr_datasets, dim='new_dim')

ndvi_combined = xr.concat(ndvi_datasets, dim='new_dim')
evi_combined = xr.concat(evi_datasets, dim='new_dim')

day_average = day_combined.mean(dim='new_dim', skipna=True)
night_average = night_combined.mean(dim='new_dim', skipna=True)
dtr_average = dtr_combined.mean(dim='new_dim', skipna=True)

ndvi_average = ndvi_combined.mean(dim='new_dim', skipna=True)
evi_average = evi_combined.mean(dim='new_dim', skipna=True)


test = day_average - night_average
#day_average.plot()
#night_average.plot()
#dtr_average.plot()


eco = gpd.read_file("data/sandhills_shapefile/us_eco_l4_no_st.shp")
sandhills = eco[eco["US_L4CODE"] == "65c"]
sandhills = sandhills.to_crs(OUTPUT_CRS)


def calculate_mean(lst, sandhills):
    sandhills = sandhills.dissolve()

    # Calculate average temp within sandhills region
    inside = lst.rio.clip(sandhills.geometry, sandhills.crs, drop = True)
    inside_mean = inside.mean().item()
    
    # Calculate average temp within a 75 km buffer around the sandhills region
    buffered = sandhills.buffer(75000) # meter
    background = gpd.GeoDataFrame(geometry=buffered, crs=sandhills.crs)
    
    outside = lst.rio.clip(background.geometry, background.crs, drop=True)
    outside = outside.where(
        ~outside.rio.clip(sandhills.geometry, sandhills.crs).notnull()
    )
    
    outside_mean = outside.mean().item()
    
    return inside_mean, outside_mean, background


def plot_lst(lst, sandhills, plot_type, save = False):
    '''
    
    @param lst averaged lst dataframe generated above
    @param sandhills shapefile processed above
    @plot_type either 'day', 'night', or 'range' range is the diurnal range, day - night
    @save defaults to false - set to true to save to figures folder
    '''
    proj = ccrs.UTM(zone=17)
    
    lonW = -83.100586
    latS = 32.082575
    lonE = -75.563965
    latN = 36.589068
    
    fig, ax = plt.subplots(
        figsize=(10, 8),
        subplot_kw={"projection": proj}
    )
    # plot raster
    temp = lst.plot(
        ax = ax,
        transform = proj,
        cmap = 'coolwarm',
        add_colorbar = False
    )
    
    cbar = plt.colorbar(
        temp,
        ax=ax,
        orientation="vertical",
        fraction=0.046,   # controls width
        pad=0.04,         # spacing from map
        shrink=0.85       # controls height
    )
    
    if plot_type == 'range':
        cbar.set_label("Diurnal Temperature Range (°C)", fontsize = 15)
    else:
        cbar.set_label("Land Surface Temperature (°C)", fontsize = 15)
    
    cbar.ax.tick_params(labelsize=13)
    
    ax.add_geometries(
        sandhills.geometry,
        crs=proj,
        facecolor='none',
        edgecolor='black',
        linewidth=1.5
    )
    
    ax.set_extent([lonW,lonE,latS,latN], crs=ccrs.PlateCarree())
    # add boundaries
    ax.add_feature(cfeature.STATES, linewidth=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    
    
    if plot_type == 'day':
        ax.set_title("Averaged Daytime JJA LST (shaded)", fontsize = 20)
    elif plot_type == 'night':
        ax.set_title("Averaged Nighttime JJA LST (shaded)", fontsize = 20)
    elif plot_type == 'range':
        ax.set_title("JJA LST Diurnal Range (shaded)\n[Mean Daytime] - [Mean Nighttime]", fontsize = 20)
    else:
        ax.set_title("")
    
    # county lines
    counties = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_2_counties',
        scale='10m',
        facecolor='none'
    )
    
    ax.add_feature(counties, linewidth=0.3)
    
    inside_mean, outside_mean, background = calculate_mean(lst, sandhills)
    
    ax.add_geometries(
        background.geometry,
        crs = proj,
        facecolor = 'none',
        edgecolor = 'black',
        linewidth = 1.5,
        linestyle = '--'
    )
    
    legend_handles = [
        Line2D([0], [0], color='black', linewidth=1.5, linestyle='-',
               label='Sandhills boundary'),
        Line2D([0], [0], color='black', linewidth=1.5, linestyle='--',
               label='Control region')
    ]

    ax.legend(
        handles=legend_handles,
        loc= (.655, .12),
        fontsize = 13,
        frameon=True,
        facecolor='lightgrey',
        edgecolor='black'
    )
    
    ax.text(
    0.5, 0.02,
        f"Sandhills mean temperature: {inside_mean:.2f}°C\nControl mean temperature: {outside_mean:.2f}°C",
        transform = ax.transAxes,
        fontsize = 13,
        verticalalignment = 'bottom',
        horizontalalignment = 'left',
        bbox=dict(facecolor = 'lightgrey', alpha=0.7, edgecolor='black')
    )
    
    if save:
        plt.savefig(f"figures/{plot_type}_LST_map.png")
    
    plt.show()
    
    
def plot_ndvi(ndvi, sandhills, plot_type, save = False):
    '''
    
    @param ndvi averaged ndvi dataframe generated above
    @param sandhills shapefile processed above
    @plot_type either 'ndvi', 'evi'
    @save defaults to false - set to true to save to figures folder
    '''
    proj = ccrs.UTM(zone=17)
    
    lonW = -83.100586
    latS = 32.082575
    lonE = -75.563965
    latN = 36.589068
    
    fig, ax = plt.subplots(
        figsize=(10, 8),
        subplot_kw={"projection": proj}
    )
    # plot raster
    temp = ndvi.plot(
        ax = ax,
        transform = proj,
        cmap = 'RdYlGn',
        add_colorbar = False,
        vmin = 0,
        vmax = 1
    )
    
    cbar = plt.colorbar(
        temp,
        ax=ax,
        orientation="vertical",
        fraction=0.046,   # controls width
        pad=0.04,         # spacing from map
        shrink=0.85       # controls height
    )
    
    
    
    if plot_type == 'ndvi':
        cbar.set_label("NDVI", fontsize = 15)
    else:
        cbar.set_label("EVI", fontsize = 15)
        
    cbar.ax.tick_params(labelsize=13)
    
    ax.add_geometries(
        sandhills.geometry,
        crs=proj,
        facecolor='none',
        edgecolor='black',
        linewidth=1.5
    )
    
    ax.set_extent([lonW,lonE,latS,latN], crs=ccrs.PlateCarree())
    # add boundaries
    ax.add_feature(cfeature.STATES, linewidth=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    
    
    if plot_type == 'ndvi':
        ax.set_title("Averaged JJA NDVI (shaded)", fontsize = 20)
    elif plot_type == 'evi':
        ax.set_title("Averaged JJA EVI (shaded)", fontsize = 20)
    else:
        ax.set_title("")
    
    # county lines
    counties = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_2_counties',
        scale='10m',
        facecolor='none'
    )
    
    ax.add_feature(counties, linewidth=0.3)
    
    inside_mean, outside_mean, background = calculate_mean(ndvi, sandhills)
    
    ax.add_geometries(
        background.geometry,
        crs = proj,
        facecolor = 'none',
        edgecolor = 'black',
        linewidth = 1.5,
        linestyle = '--'
    )
    
    legend_handles = [
        Line2D([0], [0], color='black', linewidth=1.5, linestyle='-',
               label='Sandhills boundary'),
        Line2D([0], [0], color='black', linewidth=1.5, linestyle='--',
               label='Control region')
    ]

    ax.legend(
        handles=legend_handles,
        loc= (.651, .12),
        fontsize = 13,
        frameon=True,
        facecolor='lightgrey',
        edgecolor='black'
    )
    
    
    ax.text(
        0.64, 0.02,
        f"Sandhills mean {plot_type.upper()}: {inside_mean:.2f}\nControl mean {plot_type.upper()}: {outside_mean:.2f}",
        transform = ax.transAxes,
        fontsize = 13,
        verticalalignment = 'bottom',
        horizontalalignment = 'left',
        bbox=dict(facecolor = 'lightgrey', alpha=0.7, edgecolor='black')
    )
    
    if save:
        plt.savefig(f"figures/{plot_type.upper()}_map.png")
    
    plt.show()
    
    
#plot_ndvi(evi_average, sandhills, 'evi', save = True)
plot_lst(night_average, sandhills, 'night', save = True)









