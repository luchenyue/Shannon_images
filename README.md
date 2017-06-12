# Shannon_images

Image processing code for Shannon's project that compares cell shape metrics before and after freezing treatment.

Workflow:
1. extract 1 extract 10x images ✓
2. sort and group images and generate dataframes based on expt_date, condition, color_channel, device_number, and position ✓
3. cut out top or bottom where there is scalebar based on bf ✓
4. identify lanes based on bf ✓
5. make dapi masks to identify cells ✓
6. count cells based on dapi and get dapi intensity ✓
7. count dead cells based on dapi mask and red cells (viability)✓ 
8. get shape metrics (min/major axis, aspect ratio), percent coverage based on green in channel, green intensity ✓
9. batch process images and data analysis ✓
