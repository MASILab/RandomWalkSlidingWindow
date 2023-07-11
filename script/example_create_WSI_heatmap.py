import cv2
import numpy as np
import math
import sys

# The goal is to create a WSI-wise heatmap after finding the potential shifting
# Here, we assume the image patch is 1024, and the min step for shifting is 256.
# There are in total 21 potential shifting

sample_id = sys.argv[1]
def create_weighted_heatmap(sample_id,tmp_x,tmp_y,w,h,x_final_merge):
    print(f'{sample_id}_shiftX_{tmp_x}_shiftY_{tmp_y}')
    x_final_merge_tmp = np.pad(x_final_merge, ((tmp_y,0),(tmp_x,0)), 'constant')
    x_final_merge_tmp = x_final_merge_tmp[0:h, 0:w]
    np.savez_compressed(f'WSI_1024_shiftX_{tmp_x}_shiftY_{tmp_y}/{sample_id}_weighted_shiftX_{tmp_x}_shiftY_{tmp_y}.npz',x_final_merge_tmp)

        
img = np.zeros((1024,1024))
for i in range(-512,512):
    for j in range(-512,512):
        if i ==511 and j == 0:
            print(min(abs(i-0),abs(j-0)))
        img[j,i]=min(abs(i-0),abs(j-0))

img = (1-0.5)*(img-np.min(img))/(np.max(img)-np.min(img)) + 0.5

# should be the input of WSI
dapi = cv2.imread('%s_DAPI.tif' % sample_id,cv2.IMREAD_GRAYSCALE)
h,w = dapi.shape

x_v = []
for i in range(0,math.ceil(w/1024)):
    x_v.append(img)

x_v_merge = np.hstack(x_v)
x_v = []

for i in range(0,math.ceil(h/1024)):
    x_v.append(x_v_merge)

x_final_merge = np.vstack(x_v)
x_final_merge = x_final_merge[0:h, 0:w]




# 24 potential shifting points for the heat map
create_weighted_heatmap(sample_id,0,0,w,h,x_final_merge)
create_weighted_heatmap(sample_id,0,256,w,h,x_final_merge)
create_weighted_heatmap(sample_id,0,512,w,h,x_final_merge)
create_weighted_heatmap(sample_id,0,768,w,h,x_final_merge)
create_weighted_heatmap(sample_id,256,0,w,h,x_final_merge)
create_weighted_heatmap(sample_id,256,256,w,h,x_final_merge)
create_weighted_heatmap(sample_id,256,512,w,h,x_final_merge)
create_weighted_heatmap(sample_id,256,768,w,h,x_final_merge)
create_weighted_heatmap(sample_id,512,0,w,h,x_final_merge)
create_weighted_heatmap(sample_id,512,256,w,h,x_final_merge)
create_weighted_heatmap(sample_id,512,512,w,h,x_final_merge)
create_weighted_heatmap(sample_id,512,768,w,h,x_final_merge)
create_weighted_heatmap(sample_id,768,0,w,h,x_final_merge)
create_weighted_heatmap(sample_id,768,256,w,h,x_final_merge)
create_weighted_heatmap(sample_id,768,512,w,h,x_final_merge)
create_weighted_heatmap(sample_id,768,768,w,h,x_final_merge)
create_weighted_heatmap(sample_id,128,128,w,h,x_final_merge)
create_weighted_heatmap(sample_id,384,384,w,h,x_final_merge)
create_weighted_heatmap(sample_id,640,640,w,h,x_final_merge)
create_weighted_heatmap(sample_id,896,896,w,h,x_final_merge)
