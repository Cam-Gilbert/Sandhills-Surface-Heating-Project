'''
MUST USE CONDA ENVIRONMENT sandhills_lst_env
'''
from pathlib import Path
from collections import defaultdict
import numpy as np
from pyhdf.SD import SD, SDC
from rasterio.crs import CRS
from PreprocessData import parse_filename
from PreprocessData import save_geotiff
from PreprocessData import trim_files
from PreprocessData import reproject_data
from PreprocessData import read_ndvi_hdf
from PreprocessData import ndvi_quality_mask
from PreprocessData import convert_ndvi


' GLOBAL CONSTANTS '
STUDY_BOUNDS = {
    "west":-91.252441,"east": -70.345459,
    "south": 30.439202,"north": 38.401949,
}


DATA_ROOT = Path("./data")
OUTPUT_DIR = Path("./data/processed")

OUTPUT_CRS = CRS.from_epsg(32617)

# fixed tile extent - found this online i hope it works
TILE_XMIN = -7783653.638
TILE_XMAX = -6671703.118
TILE_YMIN =  3335851.559
TILE_YMAX =  4447802.078
TILE_ROWS  = 2400
TILE_COLS  = 2400
OUTPUT_CRS = CRS.from_epsg(32617)

#sinusoidal CRS - found this online i hope it works
MODIS_SIN_CRS = CRS.from_proj4("+proj=sinu +lon_0=0 +x_0=0 +y_0=0 "
    "+a=6371007.181 +b=6371007.181 +units=m +no_defs")


# Get file paths
hdf_files = sorted(DATA_ROOT.rglob("*.hdf")) # Get all hdf files only


# split lst & ndvi files into separate lists
lst_files = []
ndvi_files = []
for f in hdf_files:
    if f.name.startswith(("MOD11A1", "MYD11A1")):
        lst_files.append(f)
    else:
        ndvi_files.append(f)
        
ndvi_composites = defaultdict(list)
evi_composites = defaultdict(list)
saved_transform = None
saved_crs = None

for i, hdf_path in enumerate(ndvi_files):
    try:
        # get file & read it
        satellite, date = parse_filename(hdf_path)
        ndvi, evi, quality = read_ndvi_hdf(hdf_path)
        
        # apply cloud cover & quality masks
        ndvi = ndvi_quality_mask(ndvi, quality)
        evi = ndvi_quality_mask(evi, quality)
        
        # apply scaling factor
        ndvi = convert_ndvi(ndvi)
        evi = convert_ndvi(evi)

        # Reproject 
        ndvi, transform = reproject_data(ndvi, tile_cols = TILE_COLS, tile_rows = TILE_ROWS)
        evi, _ = reproject_data(evi, tile_cols = TILE_COLS, tile_rows = TILE_ROWS) # dont need to overwrite transform
        
        # trim
        ndvi, clip_transform = trim_files(ndvi, transform, OUTPUT_CRS)
        evi, _ = trim_files(evi, transform, OUTPUT_CRS)
        
        # save individual file
        file_stem = f"{date.year}_{date.month:}_{date.day}"
        save_geotiff(ndvi, clip_transform, OUTPUT_CRS, OUTPUT_DIR / f"ndvi_{file_stem}.tif")
        save_geotiff(evi, clip_transform, OUTPUT_CRS, OUTPUT_DIR / f"evi_{file_stem}.tif")

        # set up yearly JJA mean
        if date.month in [6, 7, 8]:
            ndvi_composites[(date.year, date.month)].append(ndvi)
            evi_composites[(date.year, date.month)].append(evi)

        
        saved_transform = clip_transform
        saved_crs = OUTPUT_CRS
        
    except Exception as e:
        print(f"Error Occured - Skipping {hdf_path.name} -- {e}")

# save mean JJA NDVI per year
for (year, month), stack in sorted(ndvi_composites.items()):
    mean_ndvi = np.nanmean(np.stack(stack, axis = 0), axis = 0)
    save_geotiff(mean_ndvi, saved_transform, saved_crs, OUTPUT_DIR / f"ndvi_jja_mean_{year}_{month:02d}.tif")
    print(f"  {year}-{month:02d}: {len(stack)} NDVI composites averaged")
    
for (year, month), stack in sorted(evi_composites.items()):
    mean_evi = np.nanmean(np.stack(stack, axis = 0), axis = 0)
    save_geotiff(mean_evi, saved_transform, saved_crs, OUTPUT_DIR / f"evi_jja_mean_{year}_{month:02d}.tif")
