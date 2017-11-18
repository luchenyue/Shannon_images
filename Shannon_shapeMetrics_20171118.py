#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 14:59:38 2017

@author: Chen
"""
#%% ----- LOAD MODULES -----
import os
# 2.5 you want to uncomment the following line if you are running the script on a server, for example Partnes Erisone cluster
#os.environ['QT_QPA_PLATFORM']='offscreen'from skimage import io
import fnmatch 
import time
import re
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from skimage.filters import sobel, sobel_h, threshold_li, threshold_yen
from skimage.morphology import dilation, disk, remove_small_holes, erosion, closing, watershed, remove_small_objects, rectangle
from skimage.measure import label, regionprops
from scipy import ndimage as ndi
from skimage.feature import corner_peaks
from skimage.color import rgb2gray, label2rgb
#%% 
def get_10x_images(input_path):
    images_list = []
    for root, dirs, files in os.walk(input_path):
        dirs[:] = [d for d in dirs if d not in input_path + 'pre_images(aquired_0621)']
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
        match_expt_date = re.search('\d+\-\d+\-\d+|pre_images',name)
        if match_expt_date:                      
            self.expt_date = match_expt_date.group()
        match_condition = re.search('(\w+)freeze', name) or re.search('(\w+)(\-freeze)',name)   
        if match_condition:                      
            self.condition = match_condition.group(1)    
        match_color_channel = re.search('bf|dp|gfp|GFP|tr|tx',name)
        if match_color_channel:                      
            self.color_channel = match_color_channel.group()    
        if self.color_channel == 'tx':
                self.color_channel = 'tr'
        if self.color_channel == 'GFP':
                self.color_channel = 'gfp'
        match_device_number = re.search('dev(\d+)',name) or re.search('device\ *(\d+)',name) or re.search('/(\d\w)',name)
        if match_device_number:                      
            self.device_number = match_device_number.group(1)           
        match_position = re.search('top_repeat|bot_repeat|top|tp|bot|bt|bottom|mid|middle|bft|bfb|dpt|dpb|gfpt|gfpb|trt|trb',name)
        if match_position:                      
            self.position = match_position.group()
            if self.position in ('tp|bft|dpt|gfpt|trt'):
                self.position = 'top'
            elif self.position in ('bt|bfb|dpb|trb|gfpb|bottom'):
                self.position = 'bot'
            elif self.position == 'middle':
                self.position = 'mid'
        else:
            self.position = 'top'
#%%
def sort(images_list):
    images_df_cols = ['Path','Expt_Date','Condition','Color_Channel','Device_Number','Position']               
    images_df = pd.DataFrame(columns=images_df_cols)
    path_list, expt_date_list, condition_list, color_channel_list, device_number_list, position_list = ([] for i in range(6))
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
    expt_date, device_number, condition, position_list, bf, dp, gfp, tr = [], [], [], [], [], [], [],[]
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
                            for color in ['bf', 'dp', 'gfp', 'tr']:
                                foo = images_df.loc[match_expt_dev_condition_position & (images_df.Color_Channel == color),'Path']
                                if len(foo) == 1:
                                    eval(color).append(images_df.Path[foo.index[0]])
                                else:
                                    eval(color).append('na')
    groups_df.Expt_Date = expt_date
    groups_df.Device_Number = device_number
    groups_df.Condition = condition
    groups_df.Position = position_list
    groups_df.BF = bf
    groups_df.Dapi = dp
    groups_df.GFP = gfp
    groups_df.TxRed = tr
    # if a set of corresponding images lack either a Dapi or GFP image, our image analysis will not apply -> delete 
    rows_to_delete = []
    for index, row in groups_df.iterrows():
        dp, gfp = str(row[5]),str(row[6])
        if len(dp) == 2 or len(gfp) == 2:
            rows_to_delete.append(index)
    groups_df = groups_df.drop(rows_to_delete).reset_index(drop=True)
    return groups_df

#%% 
def crop_out_scalebar(bf, dp, gfp, tr):
    if bf == 'na':
        bf_img = np.zeros([960,1280],dtype = bool)
    else:
        bf_img_path = input_path + '/' + bf    
        bf_img = io.imread(bf_img_path)
    dp_img_path = input_path + '/' + dp   
    dp_img = io.imread(dp_img_path)
    gfp_img_path = input_path + '/' + gfp   
    gfp_img = io.imread(gfp_img_path)   
    if tr == 'na':
        tr_img = np.zeros([960,1280],dtype = bool)
    else:
        tr_img_path = input_path + '/' + tr   
        tr_img = io.imread(tr_img_path)
    for_crop_imgs = [bf_img, dp_img, gfp_img]
    crop_min_x = 960
    crop_max_x = 0
    crop_position = "None" 
    for img in for_crop_imgs:
        if len(img.shape) == 3:
            gray_for_crop = rgb2gray(img)
        else:
            gray_for_crop = img.astype(float)/255
        if 1 in np.unique(gray_for_crop):
            labeled_img_for_crop = label(gray_for_crop)
            for region in regionprops(labeled_img_for_crop):
                [x1,y1] = region.coords[-1]
                [x2,y2] = region.coords[0]      
                if x1 < 480 and x1 > crop_max_x:
                    crop_position = 'top'
                    crop_max_x = x1
                if x2 > 480 and x2 < crop_min_x:
                    crop_position = 'bottom'                      
                    crop_min_x = x2       
    if crop_position != 'None':
        bf_cropped = rgb2gray(bf_img)[int(crop_max_x):int(crop_min_x),0:1280]  
        dp_cropped = rgb2gray(dp_img)[int(crop_max_x):int(crop_min_x),0:1280]
        gfp_cropped = rgb2gray(gfp_img)[int(crop_max_x):int(crop_min_x),0:1280]  
        tr_cropped = rgb2gray(tr_img)[int(crop_max_x):int(crop_min_x),0:1280]          
        lane_number = 7     
    else:
        bf_cropped = rgb2gray(bf_img)
        dp_cropped = rgb2gray(dp_img)
        gfp_cropped = rgb2gray(gfp_img)
        tr_cropped = rgb2gray(tr_img)
        lane_number = 8     
    return bf_cropped, dp_cropped, gfp_cropped, tr_cropped, crop_position, crop_max_x, crop_min_x, lane_number
    
#%%
def make_dp_mask(dp_cropped):
    dp_eroded = erosion(dp_cropped)
    dp_sobel = sobel(dp_eroded) 
    dp_mask = dp_sobel > (dp_sobel.max() - dp_sobel.min())/10  
    dp_closed = closing(dp_mask)
    dp_filled = ndi.binary_fill_holes(dp_closed)
    dp_small_removed = remove_small_objects(dp_filled, 100)
    dp_distance = ndi.distance_transform_edt(dp_small_removed) 
    dp_local_max = corner_peaks(dp_distance, indices=False, labels=dp_small_removed, min_distance=10)
    dp_watershed_markers = ndi.label(dp_local_max)[0]
    dp_ws = watershed(-dp_distance,dp_watershed_markers, mask=dp_small_removed)
    return dp_ws

#%% 
def make_gfp_and_cyto_mask(gfp_cropped, dp_ws):    
    dp_cyto_region = dilation(dp_ws, disk(10)) * (dp_ws == 0)
    gfp_nuc_region = gfp_cropped * (dp_ws > 0)
    gfp_cyto_region = gfp_cropped * dp_cyto_region
    return dp_cyto_region, gfp_nuc_region, gfp_cyto_region
    
#%% 
def make_tr_mask(tr_cropped):
    tr_threshold  = threshold_yen(tr_cropped) 
    tr_mask = tr_cropped > tr_threshold
    if len(np.unique(tr_mask)) != 1:
        tr_clean = remove_small_objects(tr_mask, min_size=5)
    else:
        tr_clean = np.zeros(tr_mask.shape,dtype = bool)        
    return tr_clean

#%% ----- demo on small region -----
def demo_zoom(demo_image_full_path, bf_cropped, dp_cropped, gfp_cropped, tr_cropped, dp_ws, dp_cyto_region, tr_clean):    
    fig,(ax) = plt.subplots(ncols=3, nrows=4, figsize=(30,20))
    
    # the first row shows the dp images: raw, post watershed, zoomed version of raw, zoomed version of post watershed
    sub1 = plt.subplot(3,4,1)
    sub1.imshow(dp_cropped)
    sub1.title.set_text('dp_cropped')
    sub2 = plt.subplot(3,4,2)
    color2 = label2rgb(dp_ws, image = dp_cropped, bg_label=0)
    sub2.imshow(color2)
    sub2.title.set_text('dp_ws')
    sub3 = plt.subplot(3,4,3)
    sub3.imshow(dp_cropped)
    sub3.set_ylim([400, 200])
    sub3.set_xlim([500, 800])
    sub3.title.set_text('dp_cropped_zoom')    
    sub4 = plt.subplot(3,4,4)
    color4 = label2rgb(dp_ws, image = dp_cropped, bg_label=0)
    sub4.imshow(color4)
    sub4.set_ylim([400, 200])
    sub4.set_xlim([500, 800])
    sub4.title.set_text('dp_ws_zoom') 
    
    # the second row shows the gfp images: raw, post dilation from the nuclei (donut shape), zoomed version of raw, zoomed version of post dilation
    sub5 = plt.subplot(3,4,5)
    sub5.imshow(gfp_cropped)
    sub5.title.set_text('gfp_cropped')
    sub6 = plt.subplot(3,4,6)
    color6 = label2rgb(dp_cyto_region, image = dp_cropped, bg_label=0)
    sub6.imshow(color6)
    sub6.title.set_text('gfp_cyto_region')
    sub7 = plt.subplot(3,4,7)
    sub7.imshow(gfp_cropped)
    sub7.set_ylim([400, 200])
    sub7.set_xlim([500, 800])
    sub7.title.set_text('gfp_cropped_zoom')        
    sub8 = plt.subplot(3,4,8)
    color8 = label2rgb(dp_cyto_region, image = dp_cropped, bg_label=0)
    sub8.imshow(color8)
    sub8.set_ylim([400, 200])
    sub8.set_xlim([500, 800])
    sub8.title.set_text('gfp_cyto_region_zoom') 

    # the third row shows the tr images: raw, post thresholding, zoomed version of raw, zoomed version of post thresholding
    sub9 = plt.subplot(3,4,9)
    sub9.imshow(tr_cropped)
    sub9.title.set_text('tr_cropped')
    sub10 = plt.subplot(3,4,10)
    sub10.imshow(tr_clean)
    sub10.title.set_text('tr_clean')
    sub11 = plt.subplot(3,4,11)
    sub11.imshow(tr_cropped)
    sub11.set_ylim([400, 200])
    sub11.set_xlim([500, 800])
    sub11.title.set_text('tr_cropped_zoom') 
    sub12 = plt.subplot(3,4,12)
    sub12.imshow(tr_clean)
    sub12.set_ylim([400, 200])
    sub12.set_xlim([500, 800])
    sub12.title.set_text('tr_clean_zoom') 
    
    fig.tight_layout()
    plt.savefig(demo_image_full_path)
    plt.close()
    
#%% ----- generate cell metrics -----
def generate_df(results_df_cols, image_date, image_device, image_condition, image_position, lane_numbers, dp_ws, dp_cyto_region, gfp_nuc_region, gfp_cyto_region, dp_cropped, gfp_cropped, tr_clean, csv_full_path):
    results_df_cols = ['Expt_Date', 'Device_Number','Condition','Position','Lane_Numbers','Dapi_Label','Dapi_Area','Dapi_Nuclear_Mean_Intensity','GFP_Nuclear_Mean_Intensity','GFP_Cytoplasm_Mean_Intensity','Dead_Cell']
    results_df = pd.DataFrame(columns = results_df_cols)
    expt_date, device_number, condition, position, lane_number, dapi_label, dapi_area, dapi_nuclear_mean_intensity, gfp_nuclear_mean_intensity, gfp_cytoplasm_mean_intensity, dead_cell= ([] for i in range(11))
    dp_greyscale = dp_cropped * dp_ws
    if np.array_equal([np.unique(dp_ws)],[(np.unique(dp_cyto_region))]):
        for region in regionprops(dp_ws,dp_greyscale):
            expt_date.append(image_date)
            device_number.append(image_device)
            condition.append(image_condition)
            position.append(image_position)
            lane_number.append(lane_numbers)
            dapi_label.append(region.label)
            dapi_area.append(region.area)
            dapi_nuclear_mean_intensity.append(region.mean_intensity)
        for region in regionprops(dp_ws, gfp_nuc_region.astype(int)):
            gfp_nuclear_mean_intensity.append(region.mean_intensity)
        for region in regionprops(dp_cyto_region, gfp_cyto_region.astype(int)):
            gfp_cytoplasm_mean_intensity.append(region.mean_intensity)                        
            if len(np.unique(tr_clean)) == 1:
                dead_cell.append(np.nan)
            else:
                if region.label in np.unique(tr_clean):
                    dead_cell.append(1)
                else:
                    dead_cell.append(0)
    else: 
        for region in regionprops(dp_ws,dp_greyscale):
            expt_date.append(image_date)
            device_number.append(image_device)
            condition.append(image_condition)
            position.append(image_position)
            lane_number.append(lane_numbers)
            dapi_label.append(region.label)
            dapi_area.append(region.area)
            dapi_nuclear_mean_intensity.append(region.mean_intensity)
            gfp_nuclear_mean_intensity.append(np.nan)
            gfp_cytoplasm_mean_intensity.append(np.nan)                        
            if len(np.unique(tr_clean)) == 1:
                dead_cell.append(np.nan)
            else:
                if region.label in np.unique(tr_clean):
                    dead_cell.append(1)
                else:
                    dead_cell.append(0)
                      
    results_df.Expt_Date = expt_date
    results_df.Device_Number = device_number                        
    results_df.Condition = condition
    results_df.Position = position
    results_df.Lane_Numbers = lane_number
    results_df.Dapi_Label = dapi_label
    results_df.Dapi_Area = dapi_area
    results_df.Dapi_Nuclear_Mean_Intensity = dapi_nuclear_mean_intensity
    results_df.GFP_Nuclear_Mean_Intensity = gfp_nuclear_mean_intensity    
    results_df.GFP_Cytoplasm_Mean_Intensity = gfp_cytoplasm_mean_intensity    
    results_df.Dead_Cell = dead_cell   
    if results_df.empty:
        new_row = []
        new_row.append(image_date)
        new_row.append(image_device)                       
        new_row.append(image_condition)
        new_row.append(image_position)
        for column in results_df_cols[4:]:
            new_row.append(np.nan)
        results_df = results_df.append(pd.Series(new_row, results_df_cols),ignore_index=True)
    results_df.to_csv(csv_full_path)
    return results_df_cols, results_df
#%% ----- testing on single group -----
#image_date = '3-24-17'
#image_device = '8'
#image_condition = 'post'
#image_position = 'top'
#bf = '3-24-17/dev8_SM_-10/postfreeze-10x-bf-8.tif'
#dp = '3-24-17/dev8_SM_-10/postfreeze-10x-dptop-8.tif'
#gfp = '3-24-17/dev8_SM_-10/postfreeze-10x-gfp-8.tif'
#tr = '3-24-17/dev8_SM_-10/postfreeze-10x-trtop-8.tif'

#%% ----- batch processing images -----
def process_images(groups_df):
    for index, row in groups_df.iterrows():
        bf, dp, gfp, tr = str(row[4]), str(row[5]),str(row[6]),str(row[7])
        image_date, image_device, image_condition, image_position = row[0], row[1], row[2], row[3]  
        working_directory = os.path.dirname(dp)
        output_result_folder = output_path + '/' + time.strftime("%Y%m%d")
        if not os.path.exists(output_result_folder):
            os.makedirs(output_result_folder)
        demo_image_full_path = output_result_folder + '/' + working_directory.replace('/', '_') + '_'+ image_condition + '_' + image_position + '_demo.png'
        csv_full_path = output_result_folder + '/' + working_directory.replace('/', '_')+ '_'+ image_condition + '_' + image_position + '_results_df.csv'
        bf_cropped, dp_cropped, gfp_cropped, tr_cropped, crop_position, crop_max_x, crop_min_x, lane_numbers = crop_out_scalebar(bf, dp, gfp, tr)
        print 'working on', dp 
        dp_ws = make_dp_mask(dp_cropped)
        dp_cyto_region, gfp_nuc_region, gfp_cyto_region = make_gfp_and_cyto_mask(gfp_cropped, dp_ws)
        if len(tr) > 2:
            tr_clean = make_tr_mask(tr_cropped)
        else:
            tr_cropped = np.ones(dp_cropped.shape,dtype = bool)
            tr_clean = np.ones(tr_cropped.shape,dtype = bool)
        generate_df(results_df_cols, image_date, image_device, image_condition, image_position, lane_numbers, dp_ws, dp_cyto_region, gfp_nuc_region, gfp_cyto_region, dp_cropped, gfp_cropped, tr_clean, csv_full_path)
        demo_zoom(demo_image_full_path, bf_cropped, dp_cropped, gfp_cropped, tr_cropped, dp_ws, dp_cyto_region, tr_clean)
        del bf, dp, gfp, tr, crop_position

#%% ----- add treatment details -----
# the sorting can be visualized in the flowchart I uploaded "Microvessel_grouping.png"
def match_treatment_and_delete_bad_ones(treatment_xlsx):
    treatment_df = pd.read_excel(treatment_xlsx)
    groups_to_be_analyzed = treatment_df[(treatment_df['Keep_For_Analysis(test)'] == 'Y')&( treatment_df['Figure #'] != 'N')]
    del groups_to_be_analyzed['Unnamed: 18']
    groups_to_be_analyzed=groups_to_be_analyzed.sort_values(by = ['Expt_Date', 'Device_Number','Condition','Position']).reset_index(drop=True)
    plot_groups_added = groups_to_be_analyzed.copy()    
    for index, row in plot_groups_added.iterrows():
        SM, PEG, OMG = row[8], row[9], row[10]
        if SM == 0:
            plot_group = 'No SM'
        elif SM == 1:
            if PEG == 0:
                if OMG == 0:
                    plot_group = 'SM'
                elif OMG == 50:
                    plot_group = 'SM + 3OMG (50mM)'
                elif OMG == 100:
                    plot_group = 'SM + 3OMG (100mM)'
                else:
                    print 'Something with OMG is wrong'
            elif PEG == 2:
                if OMG == 0:
                    plot_group = 'SM + PEG (2%)'
                else:
                    plot_group = 'individual plots'
            elif PEG == 5:
                if OMG == 0:
                    plot_group = 'SM + PEG (5%)'
                else:
                    plot_group = 'individual plots'          
            else: 
                print 'Something with PEG is wrong'
        else:
            print 'Something with SM is wrong'
        plot_groups_added.set_value(index,'Plot_Group',plot_group)
    return plot_groups_added

#%% 
def combine_data(output_path, plot_groups_added, results_df_cols, treatment_columns):
    clear_csvs = []
    clear_csv_df = output_path + '/' + time.strftime('%Y%m%d') + '/clear_df.csv'
    for index, row in plot_groups_added.iterrows():
        dp= str(row[5])
        image_condition = row[2]
        image_position = row[3]
        working_directory = os.path.dirname(dp)
        output_result_folder = output_path + '/' + time.strftime('%Y%m%d')
        clear_frame = pd.read_csv(output_result_folder + '/' + working_directory.replace('/', '_')+ '_'+ image_condition + '_' + image_position + '_results_df.csv')
        for treatment_column in treatment_columns:
            clear_frame[treatment_column] = [plot_groups_added[treatment_column][index]] * clear_frame.shape[0]       
        clear_csvs.append(clear_frame)
    combined_clear_csv = pd.concat(clear_csvs).reset_index(drop=True)
    combined_clear_csv.to_csv(clear_csv_df)    
    return combined_clear_csv 
    
#%%
def analyze_data(plot_groups_added, combined_clear_csv, treatment_columns):
    # please do not move the import functions to the very top. It changes the way the images look. Reason unclear, but keeping them here resolves the problem
    import seaborn as sns
    import pylab
    treatment_columns_dict = dict.fromkeys(treatment_columns)
    for treatment_column in treatment_columns:
        new_treatment_column = re.sub('[^a-zA-Z0-9 \n\.]', '', treatment_column)
        treatment_columns_dict[treatment_column] = new_treatment_column 

    total_cell_number = combined_clear_csv.groupby(['Expt_Date', 'Device_Number','Condition','Position','Lane_Numbers']).agg({'Dapi_Label': 'max'}).reset_index().rename(columns=dict(Dapi_Label='Total_Cell_Number'))
    total_cell_number['Cell_Per_Lane'] = total_cell_number['Total_Cell_Number']/total_cell_number['Lane_Numbers']
    
    death_rate = combined_clear_csv.groupby(['Expt_Date', 'Device_Number','Condition','Position','Lane_Numbers']).agg({'Dead_Cell': 'mean'}).reset_index()
    death_rate['Death_Rate'] = 100 * death_rate['Dead_Cell']
    del death_rate['Dead_Cell']    
    avg_GFP = combined_clear_csv.groupby(['Expt_Date', 'Device_Number','Condition','Position','Lane_Numbers']).agg({'GFP_Cytoplasm_Mean_Intensity': 'mean'}).reset_index().rename(columns=dict(GFP_Cytoplasm_Mean_Intensity='Average_GFP_Mean_Intensity'))

    total_and_death = pd.merge(total_cell_number,death_rate)
    all_three = pd.merge(total_and_death,avg_GFP)
    plot_groups_added['Device_Number'] = plot_groups_added['Device_Number'].astype(str)
    all_three['Device_Number'] = all_three['Device_Number'].astype(str)
    
    result_df = pd.merge(plot_groups_added, all_three, on =['Expt_Date', 'Device_Number','Condition','Position'])
    result_df_filename = output_path + '/' + time.strftime('%Y%m%d') + '/result_df.csv'
    result_df.to_csv(result_df_filename)

    # break data into groups
    SMorNoSM = result_df.loc[result_df['Plot_Group'].isin(['No SM','SM'])]
    SMPlus3OMG = result_df.loc[result_df['Plot_Group'].isin(['SM','SM + 3OMG (50mM)','SM + 3OMG (100mM)'])]
    SMPlusPEG = result_df.loc[result_df['Plot_Group'].isin(['SM','SM + PEG (2%)','SM + PEG (5%)'])]
    Ind_Data = result_df[result_df['Plot_Group'] == 'individual plots']
        
    for plot in ['Total_Cell_Number','Cell_Per_Lane','Death_Rate','Average_GFP_Mean_Intensity']:
        sns.boxplot(x='3OMG(mM)', y=plot, data= Ind_Data)   
        pylab.legend(loc='best')
        plt.savefig(output_path + '/' + time.strftime('%Y%m%d') + '/20170703_expt_' + plot + '.png')
        plt.close()  
        for data_group in ['SMorNoSM', 'SMPlus3OMG', 'SMPlusPEG']:    
            sns.boxplot(x='Plot_Group', y=plot, hue = 'temp(C)',dodge=True, data= eval(data_group))   
            pylab.legend(loc='best')
            plt.savefig(output_path + '/' + time.strftime('%Y%m%d') + '/'+ data_group + '_' + plot + '.png')
            plt.close()    
            
##%%

#%% ----- MAIN -----
# 1. change the path to where you saved your images
input_path = '/Users/Admin/Desktop/image_processing/Shannon_images/images'
# 2. change the path to where you want your output to be saved
output_path= '/Users/Admin/Desktop/image_processing/Shannon_images/results'
# 2.5 if you want to run the script on a computing server, please change the path. Please also see LOAD MODULES at the top and uncomment #2.5 there 
#input_path = '/PHShome/cl256/Python_Projects/ShannonT_CellMetrics/images'
#output_path= '/PHShome/cl256/Python_Projects/ShannonT_CellMetrics/results'
# 3. run the following two lines. Feel free to change the columns as well, but make sure you change other parts of your code accordingly
results_df_cols = ['Expt_Date', 'Device_Number','Condition','Position','Lane_Numbers','Dapi_Label','Dapi_Area','Dapi_Nuclear_Mean_Intensity','GFP_Nuclear_Mean_Intensity','GFP_Cytoplasm_Mean_Intensity','Dead_Cell']
treatment_columns = ['SM(g/L)','PEG(%)','3OMG(mM)','temp(C)','inulin(mM)','Trolox(mM)','Glycerol (%)','Trehalose(mM)','Keep_For_Analysis(test)','Figure #','Plot_Group']
# 4. run the following functions to extract, sort and process images
images_list = get_10x_images(input_path)
images_df = sort(images_list)
groups_df = group_images(images_df)
process_images(groups_df)
# 5. change the path to where you saved the excel file. In the file, you can decide whether you want to keep certain images in the analysis by entering "Y" or "N" in the "Keep_For_Analysis(test)" column 
treatment_xlsx = output_path + '/20170817/Endothelial_Microchannels-snt_copy.xlsx'
# 6. run the following functions to combine and analyze the data 
plot_groups_added = match_treatment_and_delete_bad_ones(treatment_xlsx)
combined_clear_csv = combine_data(output_path, plot_groups_added, results_df_cols, treatment_columns)    
analyze_data(plot_groups_added, combined_clear_csv, treatment_columns)
print 'done!' 
