import numpy as np
import pandas as pd
import cv2
from scipy.ndimage import center_of_mass   
from scipy.spatial import cKDTree
import tifffile
from tqdm import tqdm
import time

fragment = 0
fission = np.zeros(30)
fusion = np.zeros(30)
threshold = 2
## Initilization
file_path = "/home/jirapong/Mito_data/nellie_output/resized_control_1min_#1.ome-ch0-features_components.csv"
seg_path = "/home/jirapong/Mito_data/nellie_output/resized_control_1min_#1.ome-ch0-im_instance_label.ome.tif"
reassigned_path = "/home/jirapong/Mito_data/nellie_output/resized_control_1min_#1.ome-ch0-im_obj_label_reassigned.ome.tif"
output_name = "control"
nellie_df = pd.read_csv(file_path)
labeled_im = tifffile.imread(seg_path)
reassigned_im = tifffile.imread(reassigned_path)

first_frame_label =  len(nellie_df['reassigned_label_raw'].unique())
max_frame_num = int(nellie_df['t'].max()) + 1

final_fission_self = np.zeros((first_frame_label,max_frame_num))
final_fusion_self = np.zeros((first_frame_label,max_frame_num))
final_fission_volume = np.zeros((first_frame_label,max_frame_num))
final_fusion_volume = np.zeros((first_frame_label,max_frame_num))

final_fusion_label= np.zeros((first_frame_label,max_frame_num),dtype=object)
final_fission_label = np.zeros((first_frame_label,max_frame_num),dtype=object)


def checkThreshold_Label(df, label,threshold):
    #df and threshold matrix should have the same size. 
    thresholdMatrix = np.ones(30)
    max_frame_num = int(df['t'].max()) + 1

    for i in range(max_frame_num):
        new_label = df[df['reassigned_label_raw'] == label ]

        for j in range(threshold):
            try:
                if i < threshold:
                    new_label[new_label['t'] == i+j+1].values[0][0]
                    continue

                if i > max_frame_num-1-threshold: # check only previous 2 frame at the end of the dataset
                    #print("when I > 27")
                    new_label[new_label['t'] == i-j-1].values[0][0]
                    break
                else: #check both X frame before and after the label
                    #print(f"inside the loop when i = {i}")
                    new_label[new_label['t'] == i-j-1].values[0][0]
                    new_label[new_label['t'] == i+j+1].values[0][0]
            except IndexError:
                #print(f"Here when i = {i}")
                thresholdMatrix[i] = 0

    return thresholdMatrix

def check_Fis_Fus_Mitometer (file_path,timeDim):
    #check fission fusion from mitometer output
    file_path = file_path
    c = np.zeros((timeDim,))
    event_per_frame = np.zeros((timeDim,))
    frame_length = timeDim
    i = 0

    with open(file_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        line = line.replace('NaN', '0')
        arr = np.fromstring(line, dtype=int, sep=', ')

        if i == 0:
            c = np.stack((arr, c),axis=0)
        else: 
            c = np.vstack((c,arr))
        i += 1 
    #column =frame row =label
    # value = track number from which mitochondria fission 
    np_array = np.array(c)
    for frame in range(frame_length):
        event_per_frame[frame] = np.count_nonzero(np_array[0:,frame])

    return event_per_frame

def check_repeat(label,tree,idxs):

    temp = np.argwhere(tree[:,2] == label)
    indices = idxs[temp[0,0]][0]
    NN_arr = []

    for index in indices:
        NN = tree[index,2]
        NN_arr.append(NN)

    NN_arr = np.array(NN_arr)
    value, counts = np.unique(NN_arr,return_counts=True)
    repeated_values = value[np.argwhere(counts>1)]

    return any(NN_arr[0] == value for value in repeated_values)

def check_mito_number(df, label):
    max_frame_num = int(df['t'].max()) + 1

    for i in range(max_frame_num):

        new_label = df[df['reassigned_label_raw'] == label ]
        new_label = new_label[new_label['t'] == i]
        current_frame_fragment = new_label.shape[0]
        
        if i == 0:
            test_fragment = current_frame_fragment

        elif current_frame_fragment > test_fragment and current_frame_fragment > 0 and test_fragment > 0:
            final_fission_self[label-1][i] = final_fission_self[label-1][i] + 1
            test_fragment = current_frame_fragment 

        elif current_frame_fragment < test_fragment and current_frame_fragment > 0 and test_fragment > 0:
            final_fusion_self[label-1][i] = final_fusion_self[label-1][i] + 1
            test_fragment = current_frame_fragment
        
        elif (current_frame_fragment > test_fragment and test_fragment == 0) or (current_frame_fragment < test_fragment and current_frame_fragment == 0) or (current_frame_fragment == test_fragment):
            continue

    return final_fission_self,final_fusion_self
  
def find_extrema(binary_image):
    # Ensure the image is binary
    if binary_image.dtype != np.uint8:
        binary_image = binary_image.astype(np.uint8) * 255
        # Find contours
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    # Get the largest contour
    cnt = max(contours, key=cv2.contourArea)
    # Get bounding rectangle
    x, y, w, h = cv2.boundingRect(cnt)
    # Calculate extrema points
    leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
    rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
    topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
    bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])
    # Calculate additional points
    top_left = (x, y)
    top_right = (x + w, y)
    bottom_right = (x + w, y + h)
    bottom_left = (x, y + h)
    # Combine all points
    extrema = np.array([
        top_left,
        topmost,
        top_right,
        rightmost,
        bottom_right,
        bottommost,
        bottom_left,
        leftmost
    ])
    extrema = np.fliplr(extrema) #flip to y,x
    return extrema

def move_row_to_first(arr, value): 
    row_index = np.where((arr == value))[0] 

    if len(row_index) == 0: 
        print(f"Value {value} not found in the array.") 
        return arr 

    # Get the first occurrence if multiple matches 
    row_index = row_index[0] 
    row_of_interest = arr[row_index] 
    remaining_rows = np.delete(arr, row_index, axis=0) 
    
    return np.vstack((row_of_interest, remaining_rows)) 

def nearest_neighbour(labeled_im,csv_output,frame,neighbour):
    all_dists = []
    all_idxs = []
    centroids = []
    labels = []
    extremas = np.zeros((0,2))
    nellie_df = csv_output
    labeled_im_frame = labeled_im[frame]
    unique_label = np.unique(labeled_im_frame)
    unique_label = unique_label[unique_label != 0]

    # !!!!!cannot find a label since it always start over if we use range(len(unique_label))
    for label in unique_label:
        #find the centriod on all labels after connected component segmentation
        #center of mass function give Y then X
        #center of mass for NN element pinpoint
        centroid = center_of_mass(labeled_im_frame == label)
        centroid = np.array(list(centroid))
        centroids.append(centroid)

        #extrema for tree query
        extremas = np.concatenate((extremas,find_extrema(labeled_im_frame == label)),axis = 0)


        #find reassignment label on all label
        nellie_df_2d_label = nellie_df[nellie_df['label'] == label]
        reassigned_label = nellie_df_2d_label['reassigned_label_raw']
        labels.append(int(list(reassigned_label)[0]))
        #add number of assigned label

    centroids = np.array(centroids)
    #labels = reassigned label from the unique label. 
    labels = np.array(labels)
    labels = np.expand_dims(labels, axis=1)
    final_labels = np.repeat(labels,8,axis=0)

    extrema_treeMatrix = np.hstack((extremas,final_labels))
    treeMatrix = np.hstack((centroids,labels))

    tree = cKDTree(centroids)
    # loop again to get dist and idxs from all label
    # we need complete information of centroid to create a tree => run 2 loop
    for label in labels: 

        #centriod value of NN components
        #centriod_of_label = treeMatrix[treeMatrix[:,2]==label][:,:2]
        #avg_centriod = np.average(centriod_of_label,axis = 0)

        extrema_loc = extrema_treeMatrix[extrema_treeMatrix[:,2]==label][:,:2]

        #query k closet object
        #dists, idxs = tree.query(np.expand_dims(avg_centriod,axis = 0), k=neighbour, workers=-1)
        dists, idxs = tree.query(np.expand_dims(extrema_loc,axis = 0), k=3, workers=-1)
        dists = np.transpose(dists,(1,0,2))
        idxs = np.transpose((idxs),(1,0,2))
        all_dists.append(dists.astype(object))
        all_idxs.append(idxs.astype(object))

    all_dists = np.array(all_dists,dtype=object)
    all_idxs = np.array(all_idxs,dtype=object)

    #doesnt allow repeatition of self > 1 if happen increase neighbour and pop repeated value??
    return treeMatrix,all_dists,all_idxs,tree

def repeated_value_NN(label,NN,treeMatrix,tree,num_neighbour):

    new_NN = np.delete(NN, np.where(NN==label))
    new_NN = np.insert(new_NN,0,label)
    
    new_neighbour_needed = num_neighbour-new_NN.shape[0]

    centriod_of_label = treeMatrix[treeMatrix[:,2]==label][:,:2]
    avg_centriod = np.average(centriod_of_label,axis = 0)
    #query two closet object
    new_dists, new_idxs = tree.query(np.expand_dims(avg_centriod,axis = 0), k=num_neighbour+new_neighbour_needed, workers=-1)

    return new_dists,new_idxs

def check_volume_value(startingframe,endingframe,label,isFusion,percent_threshold = 0.15):
    volume_all = []
    volume_value = []
    checkframe = abs(endingframe- startingframe)
    if isFusion:
        for i in range(checkframe+1):
            new_label = nellie_df[nellie_df['reassigned_label_raw'] == label ]
            new_label = new_label[new_label['t']== i+startingframe]
            volume = np.array(list(new_label['area_raw']))
            volume_all.append(volume) 
    else:
          for i in range(checkframe+1):
            new_label = nellie_df[nellie_df['reassigned_label_raw'] == label ]
            new_label = new_label[new_label['t']== endingframe-i]
            volume = np.array(list(new_label['area_raw']))
            volume_all.append(volume) 
    # do we need for loop here
    for i in range(checkframe+1):
        volume_value.append(np.sum(volume_all[i]))

    arr = np.array(volume_value[0:checkframe])
    mean = np.average(arr)
   
    volume_value = np.array(volume_value)

    if np.all(volume_value>5):
        diff= (volume_value[checkframe] - mean) / mean
        significance = abs(diff) > percent_threshold
    else:
        diff = volume_value[checkframe] - mean
        significance = abs(diff)>1

    return volume_value,significance,diff

def NN_index_to_label(indices,treeMatrix):
    NN_arr = []
    for index in indices:
        NN = treeMatrix[index,2]
        NN_arr.append(NN)
    NN_arr = np.array(NN_arr)
    
    return(NN_arr)

def check_NN_volume(treeMatrix,dists,idxs,tree,label,interested_frame,isFusion):
    '''
    check the volume and distance of NN
    output = 
    column
        [0,1] : volume pre and post for fusion/ volume post and pre on fission
        [2] =  smallest distance from extrema to centriod of that label 
        [3] = label  
    
    firstrow = label of intereNNVst. 
    '''
    distance_arr = []
    temp = np.argwhere(treeMatrix[:,2] == label)
    
    #index of NN 1st column = label index, others = neighbour innex.
    #all indice in temp are equal doesnt matter which one to use
    indices = idxs[temp[0,0]]
    distance = dists[temp[0,0]]
    distance = distance[:,0,:]
    label_tobe_filtered = treeMatrix[indices.astype(np.int64)][:,0,:,2]
    NN_label = np.unique(label_tobe_filtered)

    for i in NN_label:
        filters = label_tobe_filtered==i
        minimum = np.min(distance[filters].astype(np.float128))
        distance_arr.append(minimum)

    distance_arr = np.array(distance_arr)

    volume_arr = []
    
    for index in range(NN_label.shape[0]):
        volume,_,_ = check_volume_value(interested_frame[0],interested_frame[1],NN_label[index],isFusion) #index is not a label need to get the label from tree
        volume_arr.append(volume)
    volume_arr = np.array(volume_arr)
    #insert label into the output
    final_arr = np.concatenate((volume_arr,distance_arr.reshape(distance_arr.shape[0],1),NN_label.reshape(NN_label.shape[0],1)),axis = 1)
    sorted_indices = np.argsort(final_arr[:, 0])
    final_arr = final_arr[sorted_indices]
    final_arr = move_row_to_first(final_arr, label)
    return final_arr
    
def find_combinations(arr,error_percentage=0.15,start=1,):
    '''
    find a combination of first row that could potentially sum up to the target value in second row
    return value pair data of
        1. a possible pair data of data_column to the specific_cell_value (result data)
        2. a possible pari data of target_column to the specific_cell_value (result_target)
    
        Equation:
        specific_cell_value + pre_event_arr = target_col + result_target
    '''
    pre_event_arr = []
    post_event_arr = []
    result_label = []
    result_dists = []
    #4: shape 6, type= nparr,longdoub
    #5: same, same,same
    data_col = arr[:, 0]
    target_col = arr[:, 1]
    dists_col = arr[:,2]
    label_col = arr[:,3]

    target_value = arr[0, 1]
    specific_cell_value = arr[0, 0]
    remaining_target = target_value - specific_cell_value
    sum_all_vol = target_value+specific_cell_value

    upper_boundary, lower_boundary = error_percentage*target_value, -error_percentage*target_value

    def backtrack(data_list,target_list,label_list,dist_list, start, remaining,sum,lower_bound,upper_bound):

        if (remaining/ (sum/2)) <= error_percentage:
        #remaining >= lower_bound and remaining <= upper_bound :
            #remaining inbetween this range == acceptable
            pre_event_arr.append(data_list[:])
            post_event_arr.append(target_list[:])
            result_label.append(label_list[:])
            result_dists.append(dist_list[:])
            return
        
        for i in range(start, len(data_col)):
            #second condition after and allow it to select area that only decrease
            if target_col[i] < data_col[i]:#remaining - data_col[i] >= lower_boundary and 
                data_list.append(data_col[i])
                target_list.append(target_col[i])
                label_list.append(label_col[i])
                dist_list.append(dists_col[i])
                '''
                #Above condition
                diff = (remaining - data_col[i] + target_col[i])/ (sum_all_vol/2)
                diff <= 0.15 #exit 
                #need to check how good it is 

                #for condition below
                sum_all_vol + data_col[i] + target_col[i]
                '''

                backtrack(data_list,target_list,label_list,dist_list, i + 1, remaining - (data_col[i]-target_col[i]),
                           sum + data_col[i] + target_col[i],lower_boundary, upper_boundary)
                data_list.pop()
                target_list.pop()
                label_list.pop()
                dist_list.pop()
    #when the lable suddenly appear or disappear = specific_cell_volume = 0 -> break the combination
   
    if specific_cell_value == 0: 
        return np.array([])

    else:
        backtrack([],[],[], [],start, remaining_target,sum_all_vol ,lower_boundary,upper_boundary)
        volume_pre = np.array(pre_event_arr,dtype=object)
        volume_post = np.array(post_event_arr,dtype=object)
        label = np.array(result_label,dtype=object)
        dist = np.array(result_dists,dtype=object)

    #dim are the same

    #result = np.array([volume_pre,volume_post,dist,label], dtype = object)
    result = np.transpose(np.concatenate((volume_pre,volume_post,dist,label)))

    return result
    
def gaussian_distance_weight(distance,sigma = 1.0):
    distance = distance.astype(float)
    return np.exp(-distance**2 / 2*sigma**2)

def inverse_distance_weight(distance,ratio):
    return  1/distance**ratio

def exponential_decay(distance,gamma):
    distance = distance.astype(float)
    return np.exp(-gamma*distance)

def runframe(treeMatrix,dists,idxs,tree,isFusion,label,interested_frame,fus_fiss_arr):
   
    #add observing label in another column
    volume_neighbours = check_NN_volume(treeMatrix,dists,idxs,tree,label,interested_frame,isFusion)
    combinations = find_combinations(volume_neighbours)

    if combinations.size == 0:
        return None,None
    
    #when there's only 1 combination
    try:
        if combinations.shape[1] != 0:
            section_index = int((combinations.shape[1])/4)

            pre_event_volume = combinations[:,:section_index]
            post_event_volume = combinations[:,section_index:section_index*2]
            dists = combinations[:,2*section_index:3*section_index]
            label_arr= combinations[:,3*section_index:4*section_index]
            pre_event_volume = np.array(pre_event_volume,dtype=np.float64)
            post_event_volume = np.array(post_event_volume,dtype=np.float64)
            dists = np.array(dists,dtype=np.float64)
            label_arr = np.array(label_arr,dtype=np.float64)
           
    except:
        combinations = np.concatenate(combinations)
        section_index = int(len(combinations)/4)
        pre_event_volume =  np.concatenate(np.expand_dims(combinations[:section_index],axis=0))
        post_event_volume = np.concatenate(np.expand_dims(combinations[section_index:2*section_index],axis=0))
        dists = np.concatenate(np.expand_dims(combinations[2*section_index:3*section_index],axis=0))
        label_arr = np.concatenate(np.expand_dims(combinations[3*section_index:4*section_index],axis=0))
        #post event volume for fusion, preevent volume for

    arr = np.concatenate((np.expand_dims(pre_event_volume,axis = 1),np.expand_dims(post_event_volume,axis = 1),
                          np.expand_dims(dists,axis = 1),np.expand_dims(label_arr,axis = 1)), axis = 1,dtype=np.float64)

    unique_arr = np.unique(arr,axis=0)
    if unique_arr.ndim == 3:
        unique_arr = unique_arr.transpose(1,0,2).reshape(4,-1).T
        
    weight2 = inverse_distance_weight(unique_arr[:,2],2)
    #weight = gaussian_distance_weight(unique_arr[:,0])
    #weight3 = exponential_decay(unique_arr[:,0],1)
    all_weight = np.sum(weight2)
    normalized_weight = weight2/all_weight
    event_prob = np.max(normalized_weight)

    #prob == 1 when any of the two closet neighbor ==0
    if np.any(unique_arr[:2,1] == 0):
        fus_fiss_arr[label-1][interested_frame[1]] += 1
    else:
        fus_fiss_arr[label-1][interested_frame[1]] += event_prob

    return volume_neighbours,unique_arr

def append_NN(NN,labels,frame,neighbor_arr,isFusion):
    label_NN = np.full(shape=(NN.shape[0],1) , fill_value = labels)
    
    fusion_arr = np.full(shape=(NN.shape[0],1) , fill_value = isFusion)
    if isFusion:
        frame_NN = np.full(shape=(NN.shape[0],1) , fill_value = frame)
    else:
        frame_NN = np.full(shape=(NN.shape[0],1) , fill_value = frame+1)

    NN_volume = np.concatenate((NN,fusion_arr,frame_NN,label_NN),axis = 1)
    if len(neighbor_arr) == 0:
        neighbor_arr = np.array(NN_volume)
    else:
        neighbor_arr = np.concatenate((neighbor_arr,NN_volume),axis =0) 
    return neighbor_arr

def append_event(event,labels,frame,isFusion,event_arr):
    fusion_arr = np.full(shape=(event.shape[0],1) , fill_value = isFusion)
    label_NN = np.full(shape=(event.shape[0],1) , fill_value = labels)
    if isFusion:
        frame_NN = np.full(shape=(event.shape[0],1) , fill_value = frame)
    else:
        frame_NN = np.full(shape=(event.shape[0],1) , fill_value = frame+1)
        
    fis_fus = np.concatenate((event,fusion_arr,frame_NN,label_NN),axis = 1)

    if len(event_all) == 0:
        event_arr = np.array(fis_fus)
    else:
        event_arr = np.concatenate((event_arr,fis_fus),axis =0) 
    return event_arr

treeMatrix_all = []
dists_all = []
idxs_all = []
tree_all = []
neighbour_all = []
event_all = []

for labels in tqdm(range(1,first_frame_label)):
    final_fission_self,final_fusion_self = check_mito_number(nellie_df, labels)

    for frame in range(30):
        if labels == 1:
            treeMatrix,dists,idxs,tree = nearest_neighbour(labeled_im,nellie_df,frame=frame,neighbour=6)
            treeMatrix_all.append(treeMatrix)
            dists_all.append(dists)
            idxs_all.append(idxs)
            tree_all.append(tree)

        if frame < 29:
            volume,significance,diff = check_volume_value(frame,frame +1 ,labels,isFusion = True)
            interested_frame = np.array([frame,frame+1])

            if labels == 1 and significance and np.all(volume) and diff < 0:
                isFusion = False
                treeMatrix,dists,idxs,tree = nearest_neighbour(labeled_im,nellie_df,frame=interested_frame[1],neighbour=6)
                NN_volume,event_arr  = runframe(treeMatrix,dists,idxs,tree,isFusion,labels,interested_frame,final_fission_volume)
                
                if NN_volume is not None and NN_volume.size >0  :
                    NN_volume[:,[0,1]] = NN_volume[:,[1,0]]
                    neighbour_all = append_NN(NN_volume,labels,frame,neighbour_all,isFusion)

                if event_arr is not None and event_arr.size >0  :
                    event_arr[:,[0,1]] = event_arr[:,[1,0]]
                    event_all = append_event(event_arr,labels,frame,isFusion,event_all)

            elif significance and np.all(volume):
                if diff > 0: 
                    isFusion = True
                    NN_volume,event_arr  = runframe(treeMatrix_all[frame],dists_all[frame],idxs_all[frame],tree_all[frame],
                                isFusion,labels,interested_frame,final_fusion_volume)
                                
                elif diff <0:
                    isFusion = False
                    NN_volume,event_arr  = runframe(treeMatrix_all[frame+1],dists_all[frame+1],idxs_all[frame+1],tree_all[frame+1],
                                isFusion,labels,interested_frame,final_fission_volume)
                    if NN_volume is not None:
                        NN_volume[:,[0,1]] = NN_volume[:,[1,0]]
                    if event_arr is not None:
                        event_arr[:,[0,1]] = event_arr[:,[1,0]]
                    
                if NN_volume is not None and NN_volume.size >0  :
                    neighbour_all = append_NN(NN_volume,labels,frame,neighbour_all,isFusion)

                if event_arr is not None and event_arr.size >0 :
                    event_all = append_event(event_arr,labels,frame,isFusion,event_all)
     

column_names = ["Volume_pre", "Volume_post", "Distance","Nearest Label","isFusion" ,"Frame" , "Label"]
possible_event_all = pd.DataFrame(event_all, columns=column_names)
possible_event_all.to_csv(f'{output_name}_event.csv', index=False) 

label_NN_all = pd.DataFrame(neighbour_all, columns=column_names)
label_NN_all.to_csv(f'{output_name}_neighbour.csv', index=False) 

np.savetxt(f'{output_name}_output_fission.csv',final_fission_volume, delimiter=',', fmt='%f')
np.savetxt(f'{output_name}_output_fusion.csv',final_fusion_volume, delimiter=',', fmt='%f')
np.savetxt(f'{output_name}_output_self_fission.csv',final_fission_self, delimiter=',', fmt='%f')
np.savetxt(f'{output_name}_output_self_fusion.csv',final_fusion_self, delimiter=',', fmt='%f')



