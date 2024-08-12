import scipy.ndimage
from analysis import get_nellie_inputs, get_nellie_outputs
import tifffile
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

two_dim_csv_path = "/home/jirapong/Mito_data/nellie_output/test.ome-ch0-features_components.csv"
three_dim_csv_path = "/home/jirapong/Mito_data/072720/nellie_output/experiment-79.ome-ch0-features_components.csv"
#get_nellie_inputs(raw_basename, nellie_outputs, nellie_output_files)
#result nellie_Fission_Fusion_output/adjusted_1G1M.ome-ch0-features_components.csv

def check_fission_fusion_MM(path):
    with open(path, 'r') as file:
        lines = file.readlines()
    data = []
    for line in lines:
        line = line.strip()
        row = [0 if x.lower() == 'nan' else int(x) for x in line.split(',')]
        data.append(row)
    array = np.array(data)
    event_number = np.count_nonzero(array,axis = 0)
    return event_number

nellie_fission, nellie_fusion, event_per_frame = get_nellie_outputs(two_dim_csv_path)
print(f" fission value is {nellie_fission} \n")
print(f" fusion value is {nellie_fusion} \n")

reassigned_path = "/home/jirapong/Mito_data/nellie_output/test.ome-ch0-im_obj_label_reassigned.ome.tif"
file_path = "/home/jirapong/Mito_data/nellie_output/test.ome-ch0-features_components.csv"
seg_path = "/home/jirapong/Mito_data/nellie_output/test.ome-ch0-im_instance_label.ome.tif"

fission_path_05 = "/home/jirapong/nellie/0.5_glu_output_fission.csv"
fusion_path_05 = "/home/jirapong/nellie/0.5_glu_output_fusion.csv"
fission_path_0 = "/home/jirapong/nellie/0_glu_output_fission.csv"
fusion_path_0 = "/home/jirapong/nellie/0_glu_output_fusion.csv"
fission_path_1 = "/home/jirapong/nellie/1_glu_output_fission.csv"
fusion_path_1 = "/home/jirapong/nellie/1_glu_output_fusion.csv"
fission_path_2 = "/home/jirapong/nellie/2_glu_output_fission.csv"
fusion_path_2 = "/home/jirapong/nellie/2_glu_output_fusion.csv"
fission_path_control = "/home/jirapong/nellie/control_output_fission.csv"
fusion_path_control = "/home/jirapong/nellie/control_output_fusion.csv"

fission_path_05_self = "/home/jirapong/nellie/0.5_glu_output_self_fission.csv"
fusion_path_05_self = "/home/jirapong/nellie/0.5_glu_output_self_fusion.csv"
fission_path_0_self = "/home/jirapong/nellie/0_glu_output_self_fission.csv"
fusion_path_0_self = "/home/jirapong/nellie/0_glu_output_self_fusion.csv"
fission_path_1_self = "/home/jirapong/nellie/1_glu_output_self_fission.csv"
fusion_path_1_self = "/home/jirapong/nellie/1_glu_output_self_fusion.csv"
fission_path_2_self = "/home/jirapong/nellie/2_glu_output_self_fission.csv"
fusion_path_2_self = "/home/jirapong/nellie/2_glu_output_self_fusion.csv"
fission_path_control_self = "/home/jirapong/nellie/control_output_self_fission.csv"
fusion_path_control_self = "/home/jirapong/nellie/control_output_self_fusion.csv"

fission_df_05_self = pd.read_csv(fission_path_05_self)
fusion_df_05_self = pd.read_csv(fusion_path_05_self)
fission_df_0_self = pd.read_csv(fission_path_0_self)
fusion_df_0_self = pd.read_csv(fusion_path_0_self)
fission_df_1_self = pd.read_csv(fission_path_1_self)
fusion_df_1_self = pd.read_csv(fusion_path_1_self)
fission_df_2_self = pd.read_csv(fission_path_2_self)
fusion_df_2_self = pd.read_csv(fusion_path_2_self)
fission_df_control_self = pd.read_csv(fission_path_control_self)
fusion_df_control_self = pd.read_csv(fusion_path_control_self)

fission_event_05_self = np.count_nonzero(fission_df_05_self,axis = 0)
fusion_event_05_self = np.count_nonzero(fusion_df_05_self,axis = 0)
fission_event_0_self = np.count_nonzero(fission_df_0_self,axis = 0)
fusion_event_0_self = np.count_nonzero(fusion_df_0_self,axis = 0)
fission_event_1_self = np.count_nonzero(fission_df_1_self,axis = 0)
fusion_event_1_self = np.count_nonzero(fusion_df_1_self,axis = 0)
fission_event_2_self = np.count_nonzero(fission_df_2_self,axis = 0)
fusion_event_2_self = np.count_nonzero(fusion_df_2_self,axis = 0)
fission_event_control_self = np.count_nonzero(fission_df_control_self,axis = 0)
fusion_event_control_self = np.count_nonzero(fusion_df_control_self,axis = 0)

all_fission_05 = np.sum(fission_event_05_self)
all_fusion_05 = np.sum(fusion_event_05_self)
all_fission_0 = np.sum(fission_event_0_self)
all_fusion_0= np.sum(fusion_event_0_self)
all_fission_1= np.sum(fission_event_1_self)
all_fusion_1= np.sum(fusion_event_1_self)
all_fission_2= np.sum(fission_event_2_self)
all_fusion_2= np.sum(fusion_event_2_self)
all_fission_control= np.sum(fission_event_control_self)
all_fusion_control= np.sum(fusion_event_control_self)



fission_df_05 = pd.read_csv(fission_path_05)
fusion_df_05 = pd.read_csv(fusion_path_05)
fission_df_0 = pd.read_csv(fission_path_0)
fusion_df_0 = pd.read_csv(fusion_path_0)
fission_df_1 = pd.read_csv(fission_path_1)
fusion_df_1 = pd.read_csv(fusion_path_1)
fission_df_2 = pd.read_csv(fission_path_2)
fusion_df_2 = pd.read_csv(fusion_path_2)
fission_df_control = pd.read_csv(fission_path_control)
fusion_df_control = pd.read_csv(fusion_path_control)

fission_event_05 = np.count_nonzero(fission_df_05,axis = 0)
fusion_event_05 = np.count_nonzero(fusion_df_05,axis = 0)
fission_event_0 = np.count_nonzero(fission_df_0,axis = 0)
fusion_event_0 = np.count_nonzero(fusion_df_0,axis = 0)
fission_event_1 = np.count_nonzero(fission_df_1,axis = 0)
fusion_event_1 = np.count_nonzero(fusion_df_1,axis = 0)
fission_event_2 = np.count_nonzero(fission_df_2,axis = 0)
fusion_event_2 = np.count_nonzero(fusion_df_2,axis = 0)
fission_event_control = np.count_nonzero(fission_df_control,axis = 0)
fusion_event_control = np.count_nonzero(fusion_df_control,axis = 0)

all_fission_05 = np.sum(fission_event_05)
all_fusion_05 = np.sum(fusion_event_05)
all_fission_0 = np.sum(fission_event_0)
all_fusion_0= np.sum(fusion_event_0)
all_fission_1= np.sum(fission_event_1)
all_fusion_1= np.sum(fusion_event_1)
all_fission_2= np.sum(fission_event_2)
all_fusion_2= np.sum(fusion_event_2)
all_fission_control= np.sum(fission_event_control)
all_fusion_control= np.sum(fusion_event_control)

fission_05 = "/home/jirapong/Mito_data/20240811_resized_0.5_glu_1min_#1.ome.tif_fission.txt"
fusion_05 = "/home/jirapong/Mito_data/20240811_resized_0.5_glu_1min_#1.ome.tif_fusion.txt"
fission_0 = "/home/jirapong/Mito_data/20240811_resized_0_glu_1min_#1.ome.tif_fission.txt"
fusion_0 = "/home/jirapong/Mito_data/20240811_resized_0_glu_1min_#1.ome.tif_fusion.txt"
fission_1 = "/home/jirapong/Mito_data/20240811_resized_1_glu_1min_#1.ome.tif_fission.txt"
fusion_1 = "/home/jirapong/Mito_data/20240811_resized_1_glu_1min_#1.ome.tif_fusion.txt"
fission_2 = "/home/jirapong/Mito_data/20240811_resized_2_glu_1min_#1.ome.tif_fission.txt"
fusion_2 = "/home/jirapong/Mito_data/20240811_resized_2_glu_1min_#1.ome.tif_fusion.txt"
fission_control = "/home/jirapong/Mito_data/20240811_resized_control_1min_#1.ome.tif_fission.txt"
fusion_control = "/home/jirapong/Mito_data/20240811_resized_control_1min_#1.ome.tif_fusion.txt"

fission_05_MM = check_fission_fusion_MM(fission_05)
fusion_05_MM = check_fission_fusion_MM(fusion_05)
fission_0_MM = check_fission_fusion_MM(fission_0)
fusion_0_MM = check_fission_fusion_MM(fusion_0)
fission_1_MM = check_fission_fusion_MM(fission_1)
fusion_1_MM = check_fission_fusion_MM(fusion_1)
fission_2_MM = check_fission_fusion_MM(fission_2)
fusion_2_MM = check_fission_fusion_MM(fusion_2)
fission_control_MM = check_fission_fusion_MM(fission_control)
fusion_control_MM = check_fission_fusion_MM(fusion_control)

all_fission_05_MM = np.sum(fission_05_MM)
all_fusion_05_MM = np.sum(fusion_05_MM)
all_fission_0_MM = np.sum(fission_0_MM)
all_fusion_0_MM= np.sum(fusion_0_MM)
all_fission_1_MM= np.sum(fission_1_MM)
all_fusion_1_MM= np.sum(fusion_1_MM)
all_fission_2_MM= np.sum(fission_2_MM)
all_fusion_2_MM= np.sum(fusion_2_MM)
all_fission_control_MM= np.sum(fission_control_MM)
all_fusion_control_MM= np.sum(fusion_control_MM)



with open('/home/jirapong/Mito_data/20240810_test.ome.tif_fusion.txt', 'r') as file:
    lines = file.readlines()
data = []
for line in lines:
    line = line.strip()
    row = [0 if x.lower() == 'nan' else int(x) for x in line.split(',')]
    data.append(row)
array = np.array(data)
fusion_number = np.count_nonzero(array,axis = 0)

nellie_df = pd.read_csv(file_path)
labeled_im = tifffile.imread(seg_path)
reassigned_im = tifffile.imread(reassigned_path)

