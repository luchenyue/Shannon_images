#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 14:59:38 2017

@author: Chen
"""
# looking for: shape metrics, cell count, viability (we have other images with Hoescht/PI), 
# and "percent coverage" throughout the channels. 
# Dapi and red are only available in postfreeze images 
# workflow
# step 1. extract 10x images ✓
# step 2. sort and group images and generate dataframes based on expt_date, condition, color_channel, device_number, and position ✓
# step 3. cut out top or bottom where there is scalebar based on bf ✓
# step 4. identify lanes based on bf ✓
# step 5. make dapi masks to identify cells ✓
# step 6. count cells based on dapi and get dapi intensity ✓
# step 7. count dead cells based on dapi mask and red cells (viability)✓ -> maybe expand more to capture red in nearby region 
# step 8. get shape metrics (min/major axis, aspect ratio), percent coverage based on green in channel, green intensity ✓
# step 9. batch process images and data analysis ✓
# step 10. improve segmentation algorithms (adaptive threshold? setting different seeds for watershedding)
# step 11. improve lane identifying algorithm to better identify rectangular shape and fill holes?
# limitations: 
#    a. does not process prefreeze images -> most of them do not have dapi
#    b. identify_lanes function does not work well when external light affects light distribution
#
# current summary:
#     1. able to get shape metrics when BF, Dapi, and GFP are present
#     2. able to get viability when TxRed is present
#     3. most pre-freeze experiments only have BF and GFP -> can't identify cells 
#     4. some BF images have strong background or room light -> can't identify lanes properly 
#     5. good example: /Users/Admin/Desktop/Shannon_images/3-23-17/dev3_SM_-6
#     6. bad example: /Users/Admin/Desktop/Shannon_images/3-23-17/dev2_5PEG_SM_-6 (bad dapi ws)
#                     /Users/Admin/Desktop/Shannon_images/03-03-2017/device2 (bad channel segmentation due to background light)
#     7. intermediate example: /Users/Admin/Desktop/Shannon_images/3-23-17/dev13_2PEG_SM_-10 
#%% ----- LOAD MODULES -----
import os
import fnmatch 
import re
import numpy as np
import pandas as pd
from shutil import copy
from matplotlib import pyplot as plt
from skimage.filters import threshold_otsu, sobel_h, threshold_li, threshold_yen, try_all_threshold
from skimage import io, filters
from skimage.morphology import remove_small_holes, binary_dilation, erosion, closing, watershed, remove_small_objects, rectangle
from skimage.measure import label, regionprops
from scipy import ndimage as ndi
from skimage.feature import corner_peaks
from skimage.color import rgb2gray

#%% 
def get_10x_images(input_path):
    images_list = []
    for root, dirs, files in os.walk(input_path):
        for file in files:
            match = re.search('4x|20x',file)
            if not match:
                if fnmatch.fnmatch(file, '*.tif') or fnmatch.fnmatch(file, '*.jpg') and not ('4x') in file and not ('20x') in file:
                    rel_file_path = os.path.relpath(os.path.join(root, file), input_path)
                    images_list.append(rel_file_path)
    return images_list    
#%% 
class Image:
    def __init__(self,name):
        """Return a Image object whose name is relative path.""" 
        self.name = name
        match_expt_date = re.search('\d+\-\d+\-\d+',name)
        if match_expt_date:                      
            self.expt_date = match_expt_date.group()
        match_condition = re.search('(\w+)freeze',name)   
        if match_condition:                      
            self.condition = match_condition.group(1)    
        match_color_channel = re.search('tr|tx|bf|dp|gfp|GFP',name)
        if match_color_channel:                      
            self.color_channel = match_color_channel.group()    
        if self.color_channel == 'tx':
                self.color_channel = 'tr'
        if self.color_channel == 'GFP':
                self.color_channel = 'gfp'
        match_device_number = re.search('dev(\d+)',name) or re.search('device\ *(\d+)',name)
        if match_device_number:                      
            self.device_number = match_device_number.group(1)           
        match_position = re.search('top_repeat|bot_repeat|top|tp|bot|bt|bottom|mid|middle',name)
        if match_position:                      
            self.position = match_position.group()
            if self.position == 'tp':
                self.position = 'top'
            if self.position == 'bt' or self.position == 'bottom':
                self.position = 'bot'
            if self.position == 'middle':
                self.position = 'mid'
        else:
            self.position = 'top'

#%%
def sort(images_list):
    images_df_cols = ['Path','Expt_Date','Condition','Color_Channel','Device_Number','Position']               
    images_df = pd.DataFrame(columns=images_df_cols)
    path_list = []
    expt_date_list = []
    condition_list = []
    color_channel_list = []
    device_number_list = []
    position_list = []
    for image in images_list:
        x = Image(image)
        path_list.append(x.name)
        expt_date_list.append(x.expt_date)
        condition_list.append(x.condition)
        color_channel_list.append(x.color_channel)
        device_number_list.append(x.device_number)
        position_list.append(x.position)
    images_df.Path = path_list
    images_df.Expt_Date = expt_date_list
    images_df.Condition = condition_list
    images_df.Color_Channel = color_channel_list
    images_df.Device_Number = device_number_list
    images_df.Position = position_list
    return images_df
    
#%% ----- group images based on expt, device number, and condition
def group_images(images_df):
    groups_df = pd.DataFrame(columns = ['Expt_Date', 'Device_Number', 'Condition', 'Position','BF', 'Dapi', 'GFP', 'TxRed'])
    expt_date = []
    device_number = []
    condition= []
    position_list = []
    bf = []
    dp = []
    gfp = []
    tr = []
    for expt in images_df.Expt_Date.unique():
        for dev in images_df.Device_Number.unique():
                for expt_condition in images_df.Condition.unique():
                    if len(images_df.Path[(images_df.Expt_Date == expt) & (images_df.Device_Number == dev) &(images_df.Condition == expt_condition)]) != 0:
                        for position in images_df.Position.unique():
                            expt_date.append(expt)
                            device_number.append(dev)
                            condition.append(expt_condition)
                            match_expt_dev_condition_position = (images_df.Expt_Date == expt) & (images_df.Device_Number == dev) &(images_df.Condition == expt_condition) & (images_df.Position == position)
                            position_list.append(position)
                            bf_entry = images_df.loc[match_expt_dev_condition_position & (images_df.Color_Channel == 'bf'),'Path']
                            if len(bf_entry) == 1:
                                bf.append(images_df.Path[bf_entry.index[0]])
                            else:
                                bf.append('na')
                            dp_entry = images_df.loc[match_expt_dev_condition_position & (images_df.Color_Channel == 'dp'),'Path']
                            if len(dp_entry) == 1:
                                dp.append(images_df.Path[dp_entry.index[0]])
                            else:
                                dp.append('na')
                            gfp_entry = images_df.loc[match_expt_dev_condition_position & (images_df.Color_Channel == 'gfp'),'Path']
                            if len(gfp_entry) == 1:
                                gfp.append(images_df.Path[gfp_entry.index[0]])
                            else:
                                gfp.append('na')
                            tr_entry = images_df.loc[match_expt_dev_condition_position & (images_df.Color_Channel == 'tr'),'Path']
                            if len(tr_entry) == 1:
                                tr.append(images_df.Path[tr_entry.index[0]])
                            else:
                                tr.append('na')
    groups_df.Expt_Date = expt_date
    groups_df.Device_Number = device_number
    groups_df.Condition = condition
    groups_df.Position = position_list
    groups_df.BF = bf
    groups_df.Dapi = dp
    groups_df.GFP = gfp
    groups_df.TxRed = tr
    return groups_df
    
#%% 
def clean_groups(groups_df):
    cleaned_groups = groups_df.copy()
    rows_to_delete = []
    for index, row in cleaned_groups.iterrows():
        bf, dp, gfp = str(row[4]), str(row[5]),str(row[6])
        if len(bf) == 2 or len(dp) == 2 or len(gfp) == 2:
            rows_to_delete.append(index)
    cleaned_groups = cleaned_groups.drop(rows_to_delete).reset_index(drop=True)
    return cleaned_groups
    
#%%
def random_colors():
    np.random.seed(20150929)
    colors = np.random.randint(511, size= 2000)
    return colors    
    
#%% 
def crop_out_scalebar(bf):
    bf_img_path = input_path + '/' + bf    
    bf_img = io.imread(bf_img_path)
    if len(bf_img.shape) == 3:
        bf_gray_for_crop = rgb2gray(bf_img)
    else:
        bf_gray_for_crop = bf_img.astype(float)/255
    crop_min_x = 960
    crop_max_x = 0
    crop_position = "None" 
    if 1 in np.unique(bf_gray_for_crop):
        bf_label_img_for_crop = label(bf_gray_for_crop)
        for region in regionprops(bf_label_img_for_crop):
            [x1,y1] = region.coords[-1]
            [x2,y2] = region.coords[0]      
            if x1 < 480 and x1 > crop_max_x:
                crop_position = 'top'
                crop_max_x = x1
            if x2 > 480 and x2 < crop_min_x:
                crop_position = 'bottom'                      
                crop_min_x = x2       
        if crop_position == 'top':
            bf_cropped = bf_gray_for_crop[int(crop_max_x):960,0:1280]  
        elif crop_position == 'bottom':
            bf_cropped = bf_gray_for_crop[0:int(crop_min_x),0:1280]
    else:
        bf_cropped = bf_gray_for_crop
    return bf_img, bf_cropped, crop_position, crop_max_x, crop_min_x
    
#%%
def identify_lanes(bf_img, bf_cropped):
    bf_sobel_h = sobel_h(bf_cropped)
#    fig, ax = try_all_threshold(bf_sobel_h, figsize=(15,12), verbose=False)
#    plt.show()
#    if len(bf_img.shape) == 3:
#        bf_sobel_h_threshold = threshold_otsu(bf_sobel_h)
#    else:
    bf_sobel_h_threshold = threshold_li(bf_sobel_h)
    bf_thresholded = bf_sobel_h > bf_sobel_h_threshold
    bf_closed = closing(bf_thresholded, rectangle(10,20))
    bf_dilated = binary_dilation(bf_closed)
    bf_small_holes_removed = remove_small_holes(bf_dilated,15000)
    bf_small_removed = remove_small_objects(bf_small_holes_removed)
    bf_label_image = label(bf_small_removed)
    row = bf_small_removed.shape[0]
    col = bf_small_removed.shape[1]    
    lane_mask = np.zeros([row, col],dtype = int)
    mask_label = 1
    for region in regionprops(bf_label_image):
        if region.area > 10000:
            if region.bbox[2] == 0 or region.bbox[2] == row:
                for coord in region.coords:
                    [x,y] = coord
                    lane_mask[x,y] = 0
            else:
                for coord in region.coords:
                    [x,y] = coord
                    lane_mask[x,y] = mask_label
            mask_label += 1
    lane_binary_mask = lane_mask > 0        
    return lane_mask,lane_binary_mask,  row, col
    
#%%
def make_dp_mask(dp, lane_binary_mask, crop_position, crop_max_x, crop_min_x):
    dp_img = io.imread(input_path + '/' + dp)   
    dp_gray_for_crop = rgb2gray(dp_img)
    if crop_position == 'top':
        dp_cropped = dp_gray_for_crop[int(crop_max_x):960,0:1280]  
    elif crop_position == 'bottom':
        dp_cropped = dp_gray_for_crop[0:int(crop_min_x),0:1280]
    else:
        dp_cropped = dp_gray_for_crop
    dp_eroded = erosion(dp_cropped)
    dp_eroded_threshold  = filters.threshold_li(dp_eroded) 
    dp_mask = dp_cropped > dp_eroded_threshold
    dp_threshold  = filters.threshold_li(dp_cropped) 
    dp_mask = dp_cropped > dp_threshold

    dp_distance = ndi.distance_transform_edt(dp_mask) 
    dp_local_max = corner_peaks(dp_distance, indices=False, labels=dp_mask, min_distance=10)
    dp_watershed_markers = ndi.label(dp_local_max)[0]
    dp_ws = watershed(-dp_distance,dp_watershed_markers, mask=dp_mask)
    dp_ws_lane_masked = lane_binary_mask * dp_ws 
    return dp_cropped, dp_ws_lane_masked

#%% 
def make_gfp_mask(gfp, lane_binary_mask, crop_position, crop_max_x, crop_min_x, dp_ws):    
    gfp_img = io.imread(input_path + '/' + gfp)   
    gfp_gray_for_crop = rgb2gray(gfp_img)
    if crop_position == 'top':
        gfp_cropped = gfp_gray_for_crop[int(crop_max_x):960,0:1280]  
    elif crop_position == 'bottom':
        gfp_cropped = gfp_gray_for_crop[0:int(crop_min_x),0:1280]
    else:
        gfp_cropped = gfp_gray_for_crop
    gfp_threshold = filters.threshold_yen(gfp_cropped) 
    gfp_mask = gfp_cropped > gfp_threshold
    if len(np.unique(gfp_mask)) != 1:
        gfp_distance = ndi.distance_transform_edt(gfp_mask) 
        gfp_ws = watershed(-gfp_distance, dp_ws, mask=gfp_mask)
    else:
        gfp_ws = np.zeros([row, col],dtype = bool)
    gfp_ws_lane_masked = lane_binary_mask * gfp_ws 
    return gfp_cropped, gfp_ws_lane_masked
#%% 
def make_tr_mask(tr, lane_binary_mask, crop_position, crop_max_x, crop_min_x):
    tr_img = io.imread(input_path + '/' + tr)   
    tr_gray_for_crop = rgb2gray(tr_img)
    if crop_position == 'top':
        tr_cropped = tr_gray_for_crop[int(crop_max_x):960,0:1280]  
    elif crop_position == 'bottom':
        tr_cropped = tr_gray_for_crop[0:int(crop_min_x),0:1280]
    else:
        tr_cropped = tr_gray_for_crop
    tr_threshold  = filters.threshold_yen(tr_cropped) 
    tr_mask = tr_cropped > tr_threshold
    if len(np.unique(tr_mask)) != 1:
        tr_clean = remove_small_objects(tr_mask, min_size=5)
    else:
        tr_clean = np.zeros([row, col],dtype = bool)        
    tr_clean_lane_masked = lane_binary_mask * tr_clean
    return tr_cropped, tr_clean_lane_masked

#%% ----- demo on small region -----
def demo_zoom(demo_image_full_path, bf_cropped, dp_cropped, gfp_cropped, tr_cropped, lane_mask, dp_ws, gfp_ws, tr_clean):    
    fig,(ax) = plt.subplots(ncols=4, nrows=3, figsize=(20,15))
    sub1 = plt.subplot(4,3,1)
    sub1.imshow(bf_cropped)
    sub1.title.set_text('bf_cropped')
    sub2 = plt.subplot(4,3,2)
    sub2.imshow(lane_mask)
    sub2.title.set_text('lane_mask')
    sub3 = plt.subplot(4,3,3)
    sub3.imshow(lane_mask)
    sub3.set_ylim([400, 200])
    sub3.set_xlim([500, 700])
    sub3.title.set_text('lane_mask_zoom')    
    
    sub4 = plt.subplot(4,3,4)
    sub4.imshow(dp_cropped)
    sub4.title.set_text('dp_cropped')
    sub5 = plt.subplot(4,3,5)
    sub5.imshow(dp_ws)
    sub5.title.set_text('dp_ws')
    sub6 = plt.subplot(4,3,6)
    sub6.imshow(dp_ws)
    sub6.set_ylim([400, 200])
    sub6.set_xlim([500, 700])
    sub6.title.set_text('dp_ws_zoom') 
    
    sub7 = plt.subplot(4,3,7)
    sub7.imshow(gfp_cropped)
    sub7.title.set_text('gfp_cropped')
    sub8 = plt.subplot(4,3,8)
    sub8.imshow(gfp_ws)
    sub8.title.set_text('gfp_ws')
    sub9 = plt.subplot(4,3,9)
    sub9.imshow(gfp_ws)
    sub9.set_ylim([400, 200])
    sub9.set_xlim([500, 700])
    sub9.title.set_text('gfp_ws_zoom') 
    
    sub10 = plt.subplot(4,3,10)
    sub10.imshow(tr_cropped)
    sub10.title.set_text('tr_cropped')
    sub11 = plt.subplot(4,3,11)
    sub11.imshow(tr_clean)
    sub11.title.set_text('tr_clean')
    sub12 = plt.subplot(4,3,12)
    sub12.imshow(tr_clean)
    sub12.set_ylim([400, 200])
    sub12.set_xlim([500, 700])
    sub12.title.set_text('tr_clean_zoom') 
    
    fig.tight_layout()
    plt.savefig(demo_image_full_path)
    plt.close()
    
#%% ----- generate cell metrics -----
def generate_df(lane_mask, row, col, dp_ws,gfp_ws, dp_cropped, gfp_cropped,tr_clean):
    lane_labels, lane_sizes = np.unique(lane_mask, return_counts=True)    
    results_df_cols = ['Lane_Number','Lane_Size','Dapi_Label','Dapi_Area','Dapi_Mean_Intensity','Dapi_Major_Axis_Length','Dapi_Minor_Axis_Length','GFP_Area','GFP_Mean_Intensity','GFP_Major_Axis_Length','GFP_Minor_Axis_Length','Dead_Cell']
    results_df = pd.DataFrame(columns = results_df_cols)
    lane_number = []
    lane_size = []
    dapi_label = []
    dapi_area = []
    dapi_mean_intensity = []
    dapi_major_axis_length = []
    dapi_minor_axis_length = []
    gfp_area = []
    gfp_mean_intensity = []
    gfp_major_axis_length = []
    gfp_minor_axis_length = []
    dead_cell = []
    for lane_label in lane_labels[1:]:
        individual_lane_mask = np.zeros([row, col],dtype = bool)
        for i in range(1, row):
            for j in range(1, col):
                if lane_mask[i,j] == lane_label:
                    individual_lane_mask[i,j] = True
        dp_in_lane = dp_ws * individual_lane_mask
        gfp_in_lane = gfp_ws * individual_lane_mask
        dp_in_lane_greyscale = dp_cropped * dp_ws* individual_lane_mask
        gfp_in_lane_greyscale = gfp_cropped * gfp_ws* individual_lane_mask
        if type(tr_clean) is np.ndarray:
            dead_image = dp_in_lane * tr_clean.astype(int)
        if np.array_equal([np.unique(dp_in_lane)],[(np.unique(gfp_in_lane))]):
            for region in regionprops(dp_in_lane, dp_in_lane_greyscale):
                lane_number.append(lane_label)
                lane_size.append(lane_sizes[np.where(lane_labels == lane_label)][0].astype(int))
                dapi_label.append(region.label)
                dapi_area.append(region.area)
                dapi_mean_intensity.append(region.mean_intensity)
                dapi_major_axis_length.append(region.major_axis_length)
                dapi_minor_axis_length.append(region.minor_axis_length)
                if type(tr_clean) is np.ndarray:
                    if region.label in np.unique(dead_image):
                        dead_cell.append(True)
                    else:
                        dead_cell.append(False)
                else:
                    dead_cell.append('na')
            for region in regionprops(gfp_in_lane.astype(int), gfp_in_lane_greyscale):
                gfp_area.append(region.area)
                gfp_mean_intensity.append(region.mean_intensity)
                gfp_major_axis_length.append(region.major_axis_length)
                gfp_minor_axis_length.append(region.major_axis_length)
        else: 
            for region in regionprops(dp_in_lane, dp_in_lane_greyscale):
                lane_number.append(lane_label)
                lane_size.append(lane_sizes[np.where(lane_labels == lane_label)][0].astype(int))
                dapi_label.append(region.label)
                dapi_area.append(region.area)
                dapi_mean_intensity.append(region.mean_intensity)
                dapi_major_axis_length.append(region.major_axis_length)
                dapi_minor_axis_length.append(region.minor_axis_length)
                gfp_area.append('na')
                gfp_mean_intensity.append('na')
                gfp_major_axis_length.append('na')
                gfp_minor_axis_length.append('na')
                if type(tr_clean) is np.ndarray:
                    if region.label in np.unique(dead_image):
                        dead_cell.append(True)
                    else:
                        dead_cell.append(False)
                else:
                    dead_cell.append('na')
    results_df.Lane_Number = lane_number
    results_df.Lane_Size = lane_size
    results_df.Dapi_Label = dapi_label
    results_df.Dapi_Area = dapi_area
    results_df.Dapi_Mean_Intensity = dapi_mean_intensity
    results_df.Dapi_Major_Axis_Length = dapi_major_axis_length
    results_df.Dapi_Minor_Axis_Length = dapi_minor_axis_length
    results_df.GFP_Area = gfp_area
    results_df.GFP_Mean_Intensity = gfp_mean_intensity    
    results_df.GFP_Major_Axis_Length = gfp_major_axis_length    
    results_df.GFP_Minor_Axis_Length = gfp_minor_axis_length    
    results_df.Dead_Cell = dead_cell    
    return results_df
#%% ----- testing -----
# good one
#bf = '3-31-2017_cryostage/dev8/postfreeze-10x-bfbot-8.tif'
#dp = '3-31-2017_cryostage/dev8/postfreeze-10x-dpbot-8.tif'
#tr = '3-31-2017_cryostage/dev8/postfreeze-10x-trbot-8.tif'
#gfp = '3-31-2017_cryostage/dev8/postfreeze-10x-gfpbot-8.tif'    

# bad ones
#bf = '3-23-17/dev3_SM_-6/postfreeze-10x-bftop-3.tif'
#dp = '3-23-17/dev3_SM_-6/postfreeze-10x-dptop-3.tif'
#tr = '3-23-17/dev3_SM_-6/postfreeze-10x-trtop-3.tif'
#gfp = '3-23-17/dev3_SM_-6/postfreeze-10x-gfptop-3.tif'
#
#
#bf = '03-03-2017/device3/postfreeze-10xbottom-bf-3.tif'
#dp = '03-03-2017/device3/postfreeze-10xbottom-dp-3.tif'
#tr = '03-03-2017/device3/postfreeze-10xbottom-tr-3.tif'
#gfp = '03-03-2017/device3/postfreeze-10xbottom-gfp-3.tif'

#%% ----- batch processing images -----
def process_images(cleaned_groups):
    for index, row in cleaned_groups.iterrows():
        bf, dp, gfp, tr = str(row[4]), str(row[5]),str(row[6]),str(row[7])
        image_position = row[3]
        image_condition = row[2]
        print 'working on', bf
        bf_img, bf_cropped, crop_position, crop_max_x, crop_min_x = crop_out_scalebar(bf)
        lane_mask, lane_binary_mask, row, col = identify_lanes(bf_img, bf_cropped)
        dp_cropped, dp_ws_lane_masked = make_dp_mask(dp, lane_binary_mask, crop_position, crop_max_x, crop_min_x)
        gfp_cropped, gfp_ws_lane_masked = make_gfp_mask(gfp, lane_binary_mask, crop_position, crop_max_x, crop_min_x, dp_ws_lane_masked)
        working_directory = os.path.dirname(bf)
        demo_image_full_path = input_path + '/' + working_directory + '/'+ image_condition + '_' + image_position + '_demo.png'
        csv_full_path = input_path + '/' + working_directory + '/'+ image_condition + '_' + image_position + '_results_df.csv'
        if len(tr) > 2:
            tr_cropped, tr_clean_lane_masked = make_tr_mask(tr, lane_binary_mask, crop_position, crop_max_x, crop_min_x)
        else:
            tr_cropped = np.ones([row, col],dtype = bool)
            tr_clean_lane_masked = np.ones([row, col],dtype = bool)
        results_df = generate_df(lane_mask, row, col, dp_ws_lane_masked,gfp_ws_lane_masked, dp_cropped, gfp_cropped, tr_clean_lane_masked)
        demo_zoom(demo_image_full_path, bf_cropped, dp_cropped, gfp_cropped, tr_cropped, lane_mask, dp_ws_lane_masked, gfp_ws_lane_masked, tr_clean_lane_masked)
        print 'generating', csv_full_path
        results_df.to_csv(csv_full_path)
            
#%% ----- create a repository where we can view all processed images in one folder        
def pool_demo_images(input_path):
    demo_images_list = []
    for root, dirs, files in os.walk(input_path, topdown=True):
        dirs[:] = [d for d in dirs if d not in input_path + 'images_pool']
        for file in files:
            if fnmatch.fnmatch(file, '*.png'):
                rel_file_path = os.path.relpath(os.path.join(root, file), input_path)
                demo_images_list.append(rel_file_path) 
                src_name = input_path + '/' + rel_file_path
                dst_name = input_path + '/images_pool/' + rel_file_path.replace('/', '_')
                copy(src_name, dst_name)
                
#%% ----- MAIN -----
input_path = '/Users/Admin/Desktop/Shannon_images_20170612'
images_list = get_10x_images(input_path)
images_df = sort(images_list)
groups_df = group_images(images_df)
cleaned_groups = clean_groups(groups_df)
colors = random_colors()
process_images(cleaned_groups)
pool_demo_images(input_path)
print 'done!' 