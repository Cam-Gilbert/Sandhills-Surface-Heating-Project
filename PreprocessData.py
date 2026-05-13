'''
MUST USE CONDA ENVIRONMENT sandhills_lst_env
'''
from pathlib import Path
from pyhdf.SD import SD, SDC
from shapely.geometry import box
from collections import defaultdict
import re
import pandas as pd
import numpy as np
import rasterio
import rasterio.mask
from rasterio.warp import calculate_default_transform, reproject, Resampling, transform_bounds
from rasterio.crs import CRS
from rasterio.io import MemoryFile

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
#TILE_ROWS  = 1200
#TILE_COLS  = 1200
OUTPUT_CRS = CRS.from_epsg(32617)

#sinusoidal CRS - found this online i hope it works
MODIS_SIN_CRS = CRS.from_proj4("+proj=sinu +lon_0=0 +x_0=0 +y_0=0 "
    "+a=6371007.181 +b=6371007.181 +units=m +no_defs")

# Need to parse dates from files names like this MYD11A1.A2021179.h11v05.061.2021182190406.hdf 
# A2021 is year and following 179 represents day of year - .h11 means hour 11 v05 means 5th minute. 
# I only care about the day
def parse_filename(filepath):
    '''
    Need to parse dates from files names like this MYD11A1.A2021179.h11v05.061.2021182190406.hdf 
    A2021 is year and following 179 represents day of year - .h11 means hour 11 v05 means 5th minute. 
    I only care about the day
    
    @param filepath filepath to be read
    @return Satellite name and date
    '''
    name = Path(filepath).name
    
    if name.startswith("MOD"):
        satellite = "Terra"
    else:
        satellite = "Aqua"
        
    # file parse string (i hope this works correctly)
    match = re.search(r'\.A(\d{4})(\d{3})\.', name)
    if not match:
        raise ValueError(f"Cannot parse date from file: {name}")
        
    # parse julian date into calander date
    year = int(match.group(1))
    day = int(match.group(2))
    date = pd.Timestamp(f"{year}-01-01") + pd.Timedelta(days = day - 1)
    
    return satellite, date
        
# Read hdf function.
def read_lst_hdf(hdf_path):
    '''
    Need to extract the four bands we care about from the lst hdf files
        LST_Day_1km - land surface temp field
        LST_Night_1km - likely wont be used but ill prep it just in case
        QC_Day - Quality control field
        QC_Night - likely wont be used but ill prep it just in case
        Clear_day_cov - extra quality control field to ensure no cloud cover
        Clear_night_cov - likely wont be used but ill prep it just in case
        
    @param hdf_path
    '''
    hdf = SD(str(hdf_path), SDC.READ)
    
    def extract(band):
        ds = hdf.select(band)
        data = ds[:].astype(np.float32)
        attrs = ds.attributes()
        # replace any pixels with no data with NaN
        fill = attrs.get("_FillValue", 0)
        data[data == fill] = np.nan
        return data
    
    # extract bands
    lst_day = extract("LST_Day_1km")
    lst_night = extract("LST_Night_1km")
    qc_day = extract("QC_Day").astype(np.uint8)
    qc_night = extract("QC_Night").astype(np.uint8)
    clear_day = extract("Clear_day_cov")
    clear_night = extract("Clear_night_cov")
    
    # close hdf file
    hdf.end()
    
    return lst_day, lst_night, qc_day, qc_night, clear_day, clear_night


# Cloud cover/Data Quality filter time
def cloud_cover_mask(data_array, qc_array, clear_array):
    '''
    QC Band encodes quality as a bitmask (ffs) 
    
    LST - https://ladsweb.modaps.eosdis.nasa.gov/filespec/MODIS/61/MOD11A1
        Bit field legend; uint8           (Note: bit 0 is the least significant bit)

          bit       Long name           Key

    	  1 & 0     Mandatory QA flags  00=LST produced, good quality, not
    					   necessary to examine more detailed QA
    					01=LST produced, other quality,
    					   recommend examination of more
    					   detailed QA
    					10=LST not produced due to cloud
    					   effects
    					11=LST not produced primarily due to
    					   reasons other than cloud
    
    	  2         Data Quality flag	00=good data quality of L1B in bands 31 and 32
    					01=other quality data
    	  3         Snow/Lakeice	00=no
    					01=yes according to MOD10

          5 & 4     Emis Error flag     00=average emissivity error <= 0.01
                                        01=average emissivity error <= 0.02
                                        10=average emissivity error <= 0.04
                                        11=average emissivity error >  0.04

          7 & 6    LST Error flag      00=average LST error <= 1K
                                        01=average LST error <= 2K
                                        10=average LST error <= 3K
                                        11=average LST error >  3K
                                        
    
    
    @param data_array - data array generated in read_lst_hdf 
    @param qc_array - qc data generated in read_lst_hdf 
    @param clear_array - clear data generated in read_lst_hdf 
    @return filtered lst array
    '''
    # bits 0-1 for cloud 
    qa = qc_array & 0b00000011    
    bad_quality = qa >= 2 # moderate quality >=2 change to != 0 for only the best quality
    
    
    # Also can get info on LST error so should filter for that as well
    lst_error = (qc_array >> 6) & 0b00000011 #if i did my bitwise right this should elimate any errors >2K
    bad_quality = bad_quality | (lst_error >= 3) 
    
    result = data_array.copy()
    result[bad_quality] = np.nan
    #result[clear_array < 0.5] = np.nan   # backup QA, any pizel less than 50% clear removed
    
    return result

        
# Convert temperature function
def convert_lst(lst_array):
    '''
    Raw data is in scaled integers. We need to multiple by 0.02 to move to Kelvin
    
    @param lst_array dataset containing lst
    @return converted array
    '''       
    kelvin = lst_array * 0.02
    celsius = kelvin - 273.15 
    
    return celsius
        

# Need to convert from sinusoidal projection to a standard projection
def reproject_data(data, tile_cols = 1200, tile_rows = 1200):
    ''' 
    Convert from sinusoidal project to a standard projection
    
    @oaram data dataset to be reprojected
    @param tile_cols - default is 1200 for LST data. MUST SET TO 2400 FOR NDVI PROCESSING
    @param tile_rows - default is 1200 for LST data. MUST SET TO 2400 FOR NDVI PROCESSING
    '''
    pixel_w = (TILE_XMAX - TILE_XMIN) / tile_cols
    pixel_h = (TILE_YMAX - TILE_YMIN) / tile_rows
    
    src_transform = rasterio.transform.from_origin(TILE_XMIN, TILE_YMAX, pixel_w, pixel_h)
    source_profile = {
        "driver": "GTiff", "dtype": "float32",
        "width": tile_cols, "height": tile_rows,
        "count": 1, "crs": MODIS_SIN_CRS, "transform": src_transform,
        "nodata": np.nan,
    }
    
    #do operation using MemoryFile to avoid writing
    with MemoryFile() as mem1:
        with mem1.open(**source_profile) as src:
            src.write(data[np.newaxis, :, :])
            dst_transform, dst_w, dst_h = calculate_default_transform(MODIS_SIN_CRS,
                        OUTPUT_CRS, tile_cols, tile_rows, left = TILE_XMIN, bottom = TILE_YMIN,
                        right = TILE_XMAX, top = TILE_YMAX)
            
            dst_profile = source_profile.copy()
            dst_profile.update({
                "crs": OUTPUT_CRS, "transform": dst_transform, "width": dst_w, "height": dst_h})
            
            with MemoryFile() as mem2:
                with mem2.open(**dst_profile) as dst:
                    reproject(source = rasterio.band(src, 1), 
                        destination = rasterio.band(dst, 1),
                        src_transform = src_transform,
                        src_crs = MODIS_SIN_CRS,
                        dst_transform = dst_transform,
                        dst_crs = OUTPUT_CRS,
                        resampling = Resampling.bilinear, #use nearest 4 pixels to estimate values when pixels do not fall on exact grid
                    )
                    return dst.read(1), dst_transform
        

def get_projected_bounds():
    return transform_bounds(
        "EPSG:4326",  # source CRS (lat/lon)
        OUTPUT_CRS,   # target CRS (UTM)
        STUDY_BOUNDS["west"],
        STUDY_BOUNDS["south"],
        STUDY_BOUNDS["east"],
        STUDY_BOUNDS["north"],
    )


def trim_files(data, transform, crs):
    '''
    Trims the data to just the study region of interest, saves lots of space
    
    @param data - data array
    @param transform transformation object
    @param crs coordinate reference system
    '''
    profile = {
        "driver": "GTiff", "dtype": "float32",
        "width": data.shape[1], "height": data.shape[0],
        "count": 1, "crs": crs, "transform": transform,
        "nodata": np.nan,
    }

    #geom = [box(STUDY_BOUNDS["west"], STUDY_BOUNDS["south"], STUDY_BOUNDS['east'],
    #            STUDY_BOUNDS['north']).__geo_interface__]
    
    ################## TESTING #################
    west, south, east, north = get_projected_bounds()

    geom = [box(west, south, east, north).__geo_interface__]
    ################# TESTING #################
    
    with MemoryFile() as mem:
        with mem.open(**profile) as ds:
            ds.write(data[np.newaxis, :, :])
            clipped, clip_transform = rasterio.mask.mask(ds, geom, crop = True,
                                                         nodata = np.nan, all_touched = True)
            
    return clipped[0], clip_transform

def compute_monthly_mean(monthly_day, monthly_night, output_dir, transform, crs):
    '''
    Want to get monthly averages instead of just one day, this should help get 
    better data quality and avoid single day outliars that may have been missed by QC
    
    @param monthly_day
    @param monthly_night
    @param output_dir
    @param transform
    @param crs
    
    '''
    output_dir = Path(output_dir)
    output_dir.mkdir(parents = True, exist_ok = True)
    
    for (year, month) in sorted(monthly_day.keys()):
        day_stack = np.stack(monthly_day[(year, month)], axis = 0)
        night_stack = np.stack(monthly_night[(year, month)], axis = 0)
        
        mean_day = np.nanmean(day_stack, axis = 0)
        mean_night = np.nanmean(night_stack, axis = 0)
        dtr = mean_day - mean_night
        
        # If there are less than 5 observations per pixel set to nan
        valid_counts = np.sum(~np.isnan(day_stack), axis = 0)
        for arr in [mean_day, mean_night, dtr]:
            arr[valid_counts < 2] = np.nan

        save_geotiff(mean_day, transform, crs, output_dir/f"lst_day_{year}_{month:02d}.tif")
        save_geotiff(mean_night, transform, crs, output_dir/f"lst_night_{year}_{month:02d}.tif")
        save_geotiff(dtr, transform, crs, output_dir/f"lst_dtr_{year}_{month:02d}.tif")
        
        print(f" {year}-{month:02d}: {len(monthly_day[(year, month)])} obs saved \n")


# Save files at end as a geotiff
def save_geotiff(data, transform, crs, output_path):
    '''
    Saves file after post-processing as a GeoTiff
    
    @param data data to be saved
    @transform transformation object
    @crs coordinate reference system
    @output_path - where file will be saved
    '''
    profile = {
        "driver": "GTiff", "dtype": "float32",
        "width": data.shape[1], "height": data.shape[0],
        "count": 1, "crs": crs, "transform": transform,
        "nodata": np.nan
    }
    
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(data[np.newaxis, :, :])
        
        
def read_ndvi_hdf(hdf_path):
    '''
    Need to extract the four bands we care about from the lst hdf files
        500m 16 days NDVI' - NDVI, main vegetation index
        500m 16 days EVI - likely wont be used but just in case i want an extra vegetation index
        500m 16 days pixel reliability - Quality control field
        
    @param hdf_path
    '''
    hdf = SD(str(hdf_path), SDC.READ)
    
    def extract(band):
        ds = hdf.select(band)
        data = ds[:].astype(np.float32)
        attrs = ds.attributes()
        # replace any pixels with no data with NaN
        fill = attrs.get("_FillValue", 0)
        data[data == fill] = np.nan
        return data
    
    
    ndvi = extract("500m 16 days NDVI")
    evi = extract("500m 16 days EVI") # not sure if i will use this or not
    #qa = extract("500m 16 days pixel reliability").astype(np.int8)
    qa = hdf.select("500m 16 days pixel reliability")[:].astype(np.int16)
    
    hdf.end() #close hdf
    
    return ndvi, evi, qa


def ndvi_quality_mask(data, quality):
    '''
    Mask NDVI pixels using the pixel reliability summary band.

        Pixel reliability values:
        -1 = fill / no data
        0 = good data
        1 = marginal data (produced but lower quality)
        2 = snow / ice
        3 = cloudy
    
        https://lpdaac.usgs.gov/documents/103/MOD13_User_Guide_V6.pdf

    @param data data array containing NDVI values
    @param quality array containing QA data
    @return result the masked data array
    '''
    result = data.copy()
    
    bad_qual = quality >= 2 # keep good data, lower the value to adjust sensitivity, 0 is only the best
    
    result[bad_qual] = np.nan
    return result
    

def convert_ndvi(ndvi_array):
    '''
    Raw data is in scaled integers. We need to multiple by 0.0001 to move to the real range
    
    @param ndvi_array dataset containing ndvi
    @return converted array
    '''       
    scaled = ndvi_array * 0.0001
    
    return scaled

# # Get file paths
# hdf_files = sorted(DATA_ROOT.rglob("*.hdf")) # Get all hdf files only


# # split lst & ndvi files into separate lists
# lst_files = []
# ndvi_files = []
# for f in hdf_files:
#     if f.name.startswith(("MOD11A1", "MYD11A1")):
#         lst_files.append(f)
#     else:
#         ndvi_files.append(f)





'''
                            LST Processing Loop

#Set up storage variables
monthly_day = defaultdict(list)
monthly_night = defaultdict(list)
saved_transform = None
saved_crs = None

# Iterate through files
for i, hdf_path in enumerate(lst_files):
    try:
        # get file & read it
        satellite, date = parse_filename(hdf_path)
        lst_day, lst_night, qc_day, qc_night, clear_day, clear_night = read_lst_hdf(hdf_path)
        
        # apply cloud cover & quality masks
        lst_day = cloud_cover_mask(lst_day, qc_day, clear_day)
        lst_night = cloud_cover_mask(lst_night, qc_night, clear_night)
        
        # Put into celsius
        lst_day = convert_lst(lst_day)
        lst_night = convert_lst(lst_night)

        # Reproject 
        lst_day, transform = reproject_data(lst_day)
        lst_night, _ = reproject_data(lst_night) # dont need to overwrite transform
        
        # trim
        lst_day, clip_transform = trim_files(lst_day, transform, OUTPUT_CRS)
        lst_night, _ = trim_files(lst_night, transform, OUTPUT_CRS)
        
        # build monthly lists
        monthly_day[(date.year, date.month)].append(lst_day)
        monthly_night[(date.year, date.month)].append(lst_night)
        
        saved_transform = clip_transform
        saved_crs = OUTPUT_CRS
        
    except Exception as e:
        print(f"Error Occured - Skipping {hdf_path.name} -- {e}")
        
        
# Calculate monthly averages & save
compute_monthly_mean(monthly_day, monthly_night, OUTPUT_DIR, saved_transform, saved_crs)
'''


