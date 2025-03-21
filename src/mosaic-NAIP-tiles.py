from raster_utilities import raster_mosaic, save_raster, extract_by_mask

###########################################################################
# Configurations

url = "https://s3.msi.umn.edu/stlouis-estuary-habitat/"

vector_filepath = "zip+" + f"{url}study_area_extended.zip"

# here we list filenames of NAIP tiles in order of priority
raster_filepath_list = [
    url + "naip_summer_tiles/m_4609224_nw_15_060_20220804.tif",
    url + "naip_summer_tiles/m_4609223_ne_15_060_20220804.tif",
    url + "naip_summer_tiles/m_4609223_nw_15_060_20220804.tif",
    url + "naip_summer_tiles/m_4609223_se_15_060_20220804.tif",
    url + "naip_summer_tiles/m_4609223_sw_15_060_20220804.tif",
    url + "naip_summer_tiles/m_4609224_ne_15_060_20220804.tif",
    url + "naip_summer_tiles/m_4609224_se_15_060_20220804.tif",
    url + "naip_summer_tiles/m_4609224_sw_15_060_20220804.tif",
    url + "naip_summer_tiles/m_4609222_se_15_060_20220804.tif",
    url + "naip_summer_tiles/m_4609230_ne_15_060_20220804.tif",
    url + "naip_summer_tiles/m_4609231_nw_15_060_20220804.tif",
    url + "naip_summer_tiles/m_4609117_se_15_060_20220721.tif",
    url + "naip_summer_tiles/m_4609117_nw_15_060_20220721.tif",
    url + "naip_summer_tiles/m_4609117_sw_15_060_20220721.tif",
    url + "naip_summer_tiles/m_4609117_ne_15_060_20220721.tif",
    url + "naip_summer_tiles/m_4609215_se_15_060_20210813.tif",
    url + "naip_summer_tiles/m_4609216_se_15_060_20210813.tif",
    url + "naip_summer_tiles/m_4609222_ne_15_060_20210813.tif",
    url + "naip_summer_tiles/m_4609222_sw_15_060_20210813.tif",
    url + "naip_summer_tiles/m_4609216_sw_15_060_20210813.tif"
]

# directory to save the result to
root_dir = "/home/lenkne/shared/StLouisRiver/"

###########################################################################
# Execute processes here

def main():
    print ("Mosaic NAIP tiles")
    mosaic, meta = raster_mosaic(raster_filepath_list = raster_filepath_list, nodata = 0)
    meta.update({"photometric": "RGBA"})
    print ("Save output")
    save_raster(array = mosaic, meta = meta,out_filename = root_dir + "NAIP_summer_mosaic.tif")
    print ("Extract by mask")
    cropped, meta = extract_by_mask(
        raster_filepath = root_dir + "NAIP_summer_mosaic.tif",
        vector_filepath = vector_filepath, nodata = 0)
    meta.update({"photometric" : "RGBA"})
    print ("Save output")
    save_raster(array = cropped, meta = meta, out_filename = root_dir + "NAIP_summer_mosaic_cropped.tif")
           
if __name__ == "__main__":
    main()
    
    
