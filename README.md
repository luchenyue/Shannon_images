# Shannon_images

Image processing code for Shannon's project that compares cell shape metrics before and after freezing treatment.

Workflow:
1. extract 1 extract 10x images 
2. sort and group images and generate dataframes based on expt_date, condition, color_channel, device_number, and position 
3. cut out top or bottom where there is scalebar based on bf, dp, and gfp images. If no scalebar, assume 8 lanes per image, otherwise 7 lanes.
4. make dapi masks to identify cells 
5. count cells based on dapi and get dapi intensity 
6. dilate the identified nuclei (donut shape) and get GFP intensity in that region 
7. count dead cells based on colocaliation of dapi mask and red cells (viability)
8. batch process images 
9. manually go through the processed images and take out the ones that are poorly processed and unfit for further data analysis
10. combine individual images' data ("Endothelial_Microchannels-snt_copy.xlsx")
11. sort images into groups based on treatment (for sorting procedure, please see "Microvessel_grouping.png")
12. plot data (total cell numbers, percentage of dead cells, and average GFP intensity per cell) based on the groupings 

To run the analysis, please use "Shannon_shapeMetrics_20171118.py". 

Images are too large to be uploaded to Github. 
