import cv2
import numpy as np
import glob
import sys
sample_id = sys.argv[1]
syn_name = sys.argv[2] # 1To1_noseg
merge_mode = sys.argv[3] # propose_eight

# Proposed inference with the degree of eight

f_list = []
w_list = []

# get WSI resolution

dapi = cv2.imread('%s_DAPI.tif' % sample_id,cv2.IMREAD_GRAYSCALE)
h,w = dapi.shape

shift_x_y_list = []
shift_x_y_list.append([0,0])
shift_x_y_list.append([128,128])
shift_x_y_list.append([256,256])
shift_x_y_list.append([384,384])
shift_x_y_list.append([512,512])
shift_x_y_list.append([640,640])
shift_x_y_list.append([768,768])
shift_x_y_list.append([896,896])

for shift_item in shift_x_y_list:
    shift_x = shift_item[0]
    shift_y = shift_item[1]

    f_list.append('WSI_1024_shiftX_%s_shiftY_%s/shunxing_test/results/%s_cycleGAN_a100_patch1024_%s_masked_white_background_shiftX%s_shiftY%s.tif' 
    % (shift_x, shift_y, sample_id,syn_name,shift_x, shift_y))
	
	# The weighted heat map
    dict_data_ori = np.load('WSI_1024_shiftX_%s_shiftY_%s/%s_weighted_shiftX_%s_shiftY_%s.npz' % (shift_x, shift_y,sample_id,shift_x, shift_y))
    w_list.append(dict_data_ori['arr_0'])

print(len(f_list))
sum_b = np.zeros((h, w))
sum_g = np.zeros((h, w))
sum_r = np.zeros((h, w))

# cnt = 0
w_tmp = None
for i in range (0,len(f_list)): # in f_list:
    image = cv2.imread(f_list[i])
    b,g,r = cv2.split(image)
    
    sum_b += b * w_list[i]
    sum_g += g * w_list[i]
    sum_r += r * w_list[i]
   
w_tmp = w_list[0]

# sum up all weighted
for i in range (1,len(w_list)): # in f_list:
    w_tmp = w_tmp + w_list[i]

sum_b = (sum_b/w_tmp).astype(np.uint8)
sum_g = (sum_g/w_tmp).astype(np.uint8)
sum_r = (sum_r/w_tmp).astype(np.uint8)

image_merge = cv2.merge([sum_b, sum_g, sum_r])
cv2.imwrite('WSI_weighted_average/%s_%s_%s.tif' % (sample_id, syn_name,merge_mode),image_merge)
