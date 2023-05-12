'''script by shiyu 2023-05-11
convert BEV map to Seg mask, crop raw images & mask based on ROI
'''
from google.protobuf import text_format,json_format
import os
from os import path as osp

# shiyu add: for load file
import mmcv
from projects.qcraft.datasets.proto import semantic_map_pb2
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random 
from collections import Counter

# geo
from shapely.geometry import box

# multi-process
import multiprocessing

from tqdm import tqdm


# root path
DATA_ROOT = '/mnt/vepfs/ML/ml-users/renzhe/maplearning/MapTR/data/qcraft/Map_Learning/road_geometry_intersection_autolabel/20230414_bev/bev_images/'
DATA_ROOT_WD = '/mnt/vepfs/ML/ml-users/renzhe/maplearning/MapTR/data/qcraft/Map_Learning/road_geometry_intersection_autolabel/20230414_bev/'

SAVE_ROOT = './shiyu_test_w_stoplines/'

# SAVE_MASK_ROOT = './shiyu_mask/'        # for test
SAVE_MASK_ROOT = './Bev2mask/'
SAVE_IMG_ROOT = './raw_image/'


ann_file = DATA_ROOT_WD + 'qcraft_data_infos_train_3096_filtered.pkl'

COLOR = (1, 0 , 0)
THICKNESS = 2


def parse_info_bevimage(info, data_root,  FILE_EXIST_COUNT, AREA_COUNT, rebuild=True, source_input_range = None):
    # import pdb; pdb.set_trace()
    for i in tqdm(range(len(info))):
        bevimage_path, label_path = info[i]
        label_name = bevimage_path.split('/')[-1]
        simple_id = label_name.split('_imagery.png')[0]
        label_pkl = osp.join(data_root.replace('bev_images', 'label_bev'), f'{simple_id}.pkl')

        if osp.exists(label_pkl) and not rebuild:
            # ##debug 
            # print('read from exist files', label_pkl)
            try:
                data_label = mmcv.load(label_pkl)
                return data_label
            except EOFError as e:
                print(label_pkl)

        os.makedirs(osp.dirname(label_pkl), exist_ok=True)

        semantic_map = semantic_map_pb2.SemanticMapProto()

        with open(label_path, 'r') as f:
            try:
                data = f.read()
                json_format.Parse(data, semantic_map, ignore_unknown_fields=True)
            except Exception as e:
                print("Error: ", e.__class__.__name__, e)   #continue#jia
                print(['protoerror', label_path])
                return None  

        lane_boundaries = semantic_map.lane_boundaries
        lanes = semantic_map.lanes
        stop_lines = semantic_map.lines
        
        #connect stop_lines
        stop_line_connected, stop_line_connected_type = connected_stop_line(stop_lines)

        #connected lane boundary
        lane_boundary_conneted, lane_boundary_conneted_type = connected_lane_boundary(lane_boundaries)

        #connected centerlines
        center_line_connected, center_line_connected_type = connected_centerline(lanes)

        #add by zheren crop gt label by input range
        if source_input_range is not None:
            crop_patch = box(source_input_range[0], source_input_range[1], source_input_range[2], source_input_range[3])
            lane_boundary_conneted, lane_boundary_conneted_type = crop_gt_label_by_input_range( lane_boundary_conneted, lane_boundary_conneted_type, crop_patch)
            center_line_connected, center_line_connected_type = crop_gt_label_by_input_range( center_line_connected, center_line_connected_type, crop_patch)
            stop_line_connected, stop_line_connected_type = crop_gt_label_by_input_range( stop_line_connected, stop_line_connected_type, crop_patch)
        data_label = init_sample_data(simple_id)

        # bev 坐标，中心点是原点
        data_label['lanes'] = lane_boundary_conneted
        data_label['lanes_type'] = lane_boundary_conneted_type
        data_label['centerline'] = center_line_connected
        data_label['centerline_type'] = center_line_connected_type
        data_label['stopline'] = stop_line_connected
        data_label['stopline_type'] = stop_line_connected_type
        data_label['bev_image_path'] = bevimage_path
        mmcv.dump(data_label, label_pkl)
        # draw_SegMask_on_Bev(data_label)
        # draw_SegMask_on_Bev_lanes(data_label)
        # draw_SegMask_on_Bev_stoplines(data_label)
        draw_Segmask_format(data_label, FILE_EXIST_COUNT, AREA_COUNT)
        # import pdb; pdb.set_trace()
    
    return data_label


# shiyu: draw lanes, centerlines and stoplines on raw images
def draw_SegMask_on_Bev_stoplines(data_label):
    img = cv2.imread(data_label['bev_image_path'],1)
    save_path = SAVE_ROOT  + data_label['bev_image_path'].split('/')[-1]
    raw_img = img.copy()
    # import pdb; pdb.set_trace()
    # lanes -> points (x,y)
    lanes_list = data_label['lanes']
    # h->y, w->x (2000,2000,3)
    h, w, _ = img.shape
    # mask = np.zeros((h, w))
    for i in range(len(lanes_list)):    #遍历每一条lane
        single_lane = lanes_list[i]
        # import pdb; pdb.set_trace()
        for j in range(len(single_lane)-1):
            # import pdb; pdb.set_trace()
            pt0_x, pt0_y, _ = single_lane[j]*10
            pt1_x, pt1_y, _ = single_lane[j+1]*10
            bev0 = (round(pt0_x + w/2), round(h/2 - pt0_y))
            bev1 = (round(pt1_x + w/2), round(h/2 - pt1_y))
            cv2.line(img, bev0, bev1, COLOR, THICKNESS)
            # import pdb; pdb.set_trace()
    
    stoplines_list = data_label['stopline']
    for i in range(len(stoplines_list)):    #遍历每一条lane
        single_lane = stoplines_list[i]
        # import pdb; pdb.set_trace()
        for j in range(len(single_lane)-1):
            # import pdb; pdb.set_trace()
            pt0_x, pt0_y, _ = single_lane[j]*10
            pt1_x, pt1_y, _ = single_lane[j+1]*10
            bev0 = (round(pt0_x + w/2), round(h/2 - pt0_y))
            bev1 = (round(pt1_x + w/2), round(h/2 - pt1_y))
            cv2.line(img, bev0, bev1, (0,0,255), THICKNESS)
            # import pdb; pdb.set_trace()

    centerline_list = data_label['centerline']
    for i in range(len(centerline_list)):    #遍历每一条lane
        single_lane = centerline_list[i]
        # import pdb; pdb.set_trace()
        for j in range(len(single_lane)-1):
            # import pdb; pdb.set_trace()
            pt0_x, pt0_y, _ = single_lane[j]*10
            pt1_x, pt1_y, _ = single_lane[j+1]*10
            bev0 = (round(pt0_x + w/2), round(h/2 - pt0_y))
            bev1 = (round(pt1_x + w/2), round(h/2 - pt1_y))
            cv2.line(img, bev0, bev1, (255,0,0), THICKNESS)
            # import pdb; pdb.set_trace()

    # cv2.imshow('img',img)
    result = np.hstack([raw_img, img])
    # import pdb; pdb.set_trace()
    cv2.imwrite(save_path, result)
    return None


# shiyu: visualize lanes on raw images
def draw_SegMask_on_Bev_lanes(data_label):
    img = cv2.imread(data_label['bev_image_path'],1)
    save_path = SAVE_ROOT  + data_label['bev_image_path'].split('/')[-1]
    raw_img = img.copy()
    # import pdb; pdb.set_trace()
    # lanes -> points (x,y)
    lanes_list = data_label['lanes']
    # h->y, w->x (2000,2000,3)
    h, w, _ = img.shape
    # mask = np.zeros((h, w))
    for i in range(len(lanes_list)):    #遍历每一条lane
        single_lane = lanes_list[i]
        # import pdb; pdb.set_trace()
        for j in range(len(single_lane)-1):
            # import pdb; pdb.set_trace()
            pt0_x, pt0_y, _ = single_lane[j]*10
            pt1_x, pt1_y, _ = single_lane[j+1]*10
            bev0 = (round(pt0_x + w/2), round(h/2 - pt0_y))
            bev1 = (round(pt1_x + w/2), round(h/2 - pt1_y))
            cv2.line(img, bev0, bev1, COLOR, THICKNESS)
    result = np.hstack([raw_img, img])
    cv2.imwrite(save_path, result)
    return None

def init_sample_data(simple_id):
    data = {}
    data["gt_boxes_with_id"] = {}
    data["cams"] = {}
    data["gt_boxes"] = []
    data["gt_names"] = []
    data["gt_label"] = []
    data["token"] = simple_id
    data["lidar_path"] = None
    data["sweeps"] = None
    data["timestamp"] = simple_id
    data["gt_names"] = []
    data["lane_label"] = None
    data["kpt_label"] = None
    return data

def crop_gt_label_by_input_range( lanes_conneted, lanes_conneted_type, crop_patch):

    crop_lane_connected = []
    crop_lane_type = []

    for lane, lane_type in zip(lanes_conneted, lanes_conneted_type):

        line_nodes = [(node[0], node[1])  for node in lane]

        line= LineString(line_nodes)

        if line.is_empty:  # Skip lines without nodes.
            continue
        new_line = line.intersection(crop_patch)
        if new_line.is_empty:
            continue
        else:
            if new_line.geom_type =='MultiLineString':
                for single_line in new_line.geoms:
                    single_line_lane = np.array(  [ [point_node[0], point_node[1], 0] for point_node in single_line.coords ])
                    crop_lane_connected.append(single_line_lane)
                    crop_lane_type.append(lane_type)
            else:
                new_line_lane = np.array(  [ [point_node[0], point_node[1], 0] for point_node in new_line.coords ])
                crop_lane_connected.append(new_line_lane)
                crop_lane_type.append(lane_type)
    return crop_lane_connected, crop_lane_type

def connected_lane_boundary(lane_boundaries):
    
    lane_boundary_dict = {}
    lane_boundary_visited = {}
    lane_boundary_conneted = []
    lane_boundary_conneted_type = []
    for lane_boundary in lane_boundaries:
        lane_boundary_dict[lane_boundary.id] = lane_boundary
        lane_boundary_visited[lane_boundary.id] = 0
    
    for lane_id, lane_boundary in lane_boundary_dict.items():

        if lane_boundary_visited[lane_id]:
            continue
        
        lane_boundary_visited[lane_id] = 1


        c_type =  lane_boundary.type
        prev_id = lane_boundary.start_connection_ids
        end_id = lane_boundary.end_connection_ids

        new_lane_boundary = [[point.longitude, point.latitude, 0] for point in lane_boundary.polyline.points]

        while len(prev_id)==1 and prev_id[0] in lane_boundary_dict.keys() and lane_boundary_visited[prev_id[0]]==0:
            pre_lane_boundary = lane_boundary_dict[prev_id[0]]
            new_lane_boundary = [[point.longitude, point.latitude, 0] for point in pre_lane_boundary.polyline.points] + new_lane_boundary
            lane_boundary_visited[prev_id[0]] = 1
            prev_id =  pre_lane_boundary.start_connection_ids
        
        while len(end_id) ==1 and end_id[0] in lane_boundary_dict.keys() and lane_boundary_visited[end_id[0]]==0:

            end_lane_boundary = lane_boundary_dict[end_id[0]]
            new_lane_boundary = new_lane_boundary + [[point.longitude, point.latitude, 0] for point in end_lane_boundary.polyline.points] 
            lane_boundary_visited[end_id[0]] = 1
            end_id =  end_lane_boundary.end_connection_ids
        

        if len(new_lane_boundary) ==0:
            continue

        lane_boundary_conneted.append(np.array(new_lane_boundary))
        lane_boundary_conneted_type.append(c_type)

    return lane_boundary_conneted, lane_boundary_conneted_type


def connected_centerline(lanes):
    center_line_dict = {}
    center_line_visited = {}
    center_line_connected = []
    center_line_connected_type = []
    for center_line in lanes:

        if getattr(center_line, "is_virtual", False):
            center_line.type = 4   #"VIRTUAL"
        center_line_dict[center_line.id] = center_line
        center_line_visited[center_line.id] = 0
    
    for center_line_id, center_line in center_line_dict.items():

        if center_line_visited[center_line_id]:
            continue
        
        center_line_visited[center_line_id] = 1


        c_type = center_line.type
        
        in_ids = center_line.incoming_lanes
        out_ids = center_line.outgoing_lanes
    
        new_lane_boundary = [[point.longitude, point.latitude, 0] for point in center_line.polyline.points]

        while len(in_ids)==1 and in_ids[0] in center_line_dict.keys() and center_line_visited[in_ids[0]] ==0 and center_line_dict[in_ids[0]].type == c_type:

            in_center_line = center_line_dict[in_ids[0]]
            new_lane_boundary = [[point.longitude, point.latitude, 0] for point in in_center_line.polyline.points] + new_lane_boundary
            center_line_visited[in_ids[0]] = 1
            in_ids =  in_center_line.incoming_lanes

        while len(out_ids)==1 and out_ids[0] in center_line_dict.keys() and center_line_visited[out_ids[0]] ==0 and center_line_dict[out_ids[0]].type == c_type:

            out_center_line = center_line_dict[out_ids[0]]
            new_lane_boundary = new_lane_boundary + [[point.longitude, point.latitude, 0] for point in out_center_line.polyline.points] 
            center_line_visited[out_ids[0]] = 1
            out_ids =  out_center_line.outgoing_lanes

        if len(new_lane_boundary) ==0:
            continue

        center_line_connected.append(np.array(new_lane_boundary))
        center_line_connected_type.append(c_type)
    
    return center_line_connected, center_line_connected_type


def connected_stop_line(stop_lines):
    stop_lines_conneted = []
    stop_lines_conneted_type = []

    for stop_line in stop_lines:
        new_stop_line = [[point.longitude, point.latitude, 0] for point in stop_line.polyline.points ]
        c_type =  stop_line.type

        stop_lines_conneted.append(np.array(new_stop_line))
        stop_lines_conneted_type.append(c_type)
    
    return stop_lines_conneted, stop_lines_conneted_type


# encode 
# input 2000*2000*3 -> 2000*2000*1
def rgb2id(color):
    if isinstance(color, np.ndarray) and len(color.shape) == 3:
        if color.dtype == np.uint8:
            color = color.astype(np.int32)
        return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
    return int(color[0] + 256 * color[1] + 256 * 256 * color[2])

# input 2000*2000 -> 2000*2000*3 decode
def id2rgb(id_map):
    if isinstance(id_map, np.ndarray):
        id_map_copy = id_map.copy()
        rgb_shape = tuple(list(id_map.shape) + [3])
        rgb_map = np.zeros(rgb_shape, dtype=np.uint8)
        for i in range(3):
            rgb_map[..., i] = id_map_copy % 256
            id_map_copy //= 256
        return rgb_map
    color = []
    for _ in range(3):
        color.append(id_map % 256)
        id_map //= 256
    return color


# lanes: 1000 + COUNT
# curb: 2000 + COUNT
# stoplines: 3000 + COUNT
#  np.unique()
# 新增断点功能，避免程序突然停下
# 查看路径下文件个数：  ls -l | grep "^-" | wc -l 
def draw_Segmask_format(data_label, FILE_EXIST_COUNT, AREA_COUNT):
    # 断点功能
    file_exist = os.listdir(SAVE_IMG_ROOT)
    # import pdb; pdb.set_trace()
    img = cv2.imread(data_label['bev_image_path'],0)
    save_path = SAVE_MASK_ROOT  + data_label['bev_image_path'].split('/')[-1]
    save_img_path = SAVE_IMG_ROOT + data_label['bev_image_path'].split('/')[-1]
    if data_label['bev_image_path'].split('/')[-1] in file_exist: FILE_EXIST_COUNT+=1; return

    vis_path = SAVE_MASK_ROOT  + 'map_vis_' + data_label['bev_image_path'].split('/')[-1]
    lanes_list = data_label['lanes']
    h, w = img.shape
    # lanes_type 是一串数字
    lanes_types = data_label['lanes_type']
    # 每张图片生成一个RGB的map
    sum_mask = np.zeros((h, w))
    # shiyu vis
    map_vis = np.zeros((h, w, 3))

    #定义一个只有lanes 和 stoplines 的mask
    crop_mask_sum = np.zeros((h,w)) 


    count_lanes = 0
    count_curbs = 0
    for i in range(len(lanes_list)):
        # 每条车道线有一个单独的图像，一个map是一个instance
        map = np.zeros((h, w, 3))
        
        random_r = random.randint(0,255)
        random_g = random.randint(0,255)
        random_b = random.randint(0,255)
        random_RGB = [random_r,random_g,random_b]

        single_lane = lanes_list[i]
        lane_type = lanes_types[i]

        # import pdb; pdb.set_trace()
        # shiyu: 6 for curb, check needed
        if lane_type != 6: 
            Instance_count = 1000 + count_lanes
            count_lanes += 1
            for j in range(len(single_lane)-1):
                pt0_x, pt0_y, _ = single_lane[j]*10
                pt1_x, pt1_y, _ = single_lane[j+1]*10
                bev0 = (round(pt0_x + w/2), round(h/2 - pt0_y))
                bev1 = (round(pt1_x + w/2), round(h/2 - pt1_y))
                cv2.line(map, bev0, bev1, COLOR, THICKNESS)

                mask = np.sum(map, axis=2,keepdims=False)
                mask_bool = mask > 0
                crop_mask_sum = crop_mask_sum > 0
                crop_mask_sum = crop_mask_sum | mask_bool
                # shiyu vis
                # cv2.line(map_vis, bev0, bev1, random_RGB, THICKNESS)
        # curbs
        else:
            Instance_count = 2000 + count_curbs
            count_curbs += 1
            for j in range(len(single_lane)-1):
                pt0_x, pt0_y, _ = single_lane[j]*10
                pt1_x, pt1_y, _ = single_lane[j+1]*10
                bev0 = (round(pt0_x + w/2), round(h/2 - pt0_y))
                bev1 = (round(pt1_x + w/2), round(h/2 - pt1_y))
                cv2.line(map, bev0, bev1, COLOR, THICKNESS)
                
                # mask = np.sum(map, axis=2,keepdims=False)
                # mask_bool = mask > 0
                # crop_mask_sum = crop_mask_sum > 0
                # crop_mask_sum = crop_mask_sum | mask_bool

                # shiyu vis
                # cv2.line(map_vis, bev0, bev1, random_RGB, THICKNESS)

        mask = np.sum(map, axis=2,keepdims=False)
        mask_bool = mask > 0
        # shiyu add: 判断 map 是否和当前的sum mask有交集
        sum_mask_bool = sum_mask > 0      #[0,1]
        intersection = mask_bool &  sum_mask_bool  # bool 交集的地方是0 np.unique(intersection)
        mask = mask_bool & (~intersection)
        mask = (mask>0)*Instance_count
        sum_mask += mask
        # import pdb; pdb.set_trace()
    # import pdb; pdb.set_trace()
    
    stopline_list = data_label['stopline']
    # stop_line_type = data_label['stopline_type']
    for i in range(len(stopline_list)):
        single_stopline = stopline_list[i]

        # shiyu vis
        map_vis = np.zeros((h, w, 3))
        random_r = random.randint(0,255)
        random_g = random.randint(0,255)
        random_b = random.randint(0,255)
        random_RGB = [random_r,random_g,random_b]

        # 重新生成一个map
        map = np.zeros((h, w, 3))
        Instance_count = 3000 + i
        for j in range(len(single_stopline)-1):
            pt0_x, pt0_y, _ = single_stopline[j]*10
            pt1_x, pt1_y, _ = single_stopline[j+1]*10
            bev0 = (round(pt0_x + w/2), round(h/2 - pt0_y))
            bev1 = (round(pt1_x + w/2), round(h/2 - pt1_y))
            cv2.line(map, bev0, bev1, COLOR, THICKNESS)
            
            # shiyu vis
            # cv2.line(map_vis, bev0, bev1, random_RGB, THICKNESS)

        # shiyu add crop mask:
        mask = np.sum(map, axis=2,keepdims=False)
        mask_bool = mask > 0
        crop_mask_sum = crop_mask_sum > 0
        crop_mask_sum = crop_mask_sum | mask_bool

        # import pdb; pdb.set_trace()
        mask = np.sum(map, axis=2,keepdims=False)
        mask_bool = mask > 0
        # shiyu add: 判断 map 是否和当前的sum mask有交集
        sum_mask_bool = sum_mask > 0      #[0,1]
        intersection = mask_bool &  sum_mask_bool  # bool 交集的地方是0 np.unique(intersection)
        mask = mask_bool & (~intersection)
        # mask = np.sum(map, axis=2,keepdims=False)
        mask = (mask>0)*Instance_count
        sum_mask += mask
    
    # 判断错误
    if not (sum_mask<4000).all(): import pdb; pdb.set_trace()
    # import pdb; pdb.set_trace()

    color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    color_map = id2rgb(sum_mask) # rgb->bgr opencv 存成png格式
    BGR_color_map = cv2.cvtColor(color_map, cv2.COLOR_RGB2BGR)

    # define crop mask
    row_coord, col_coord = np.where(crop_mask_sum == 1)
    # 如果coord为空，
    if len(row_coord) == 0: return
    top, down, left, right = row_coord.min(), row_coord.max(), col_coord.min(), col_coord.max()
    crop_x, crop_y, crop_width, crop_height = left, top, right - left, down - top

    if (crop_width*crop_height) < (250*500) : AREA_COUNT+=1; return
    # crop
    crop_img = color_img[ crop_y:(crop_y+crop_height), crop_x:(crop_x+crop_width),:]
    crop_mask = BGR_color_map[ crop_y:(crop_y+crop_height), crop_x:(crop_x+crop_width),:]

    # save img & mask
    cv2.imwrite(save_img_path, crop_img)
    # cv2.imwrite(save_path, crop_mask)
    
    # shiyu vis
    # cv2.imwrite(vis_path, map_vis)

    # import pdb; pdb.set_trace()
    return sum_mask




LOAD_INTERVAL = 1
def load_annotations(ann_file):
    """Load annotations from ann_file.

    Args:
        ann_file (str): Path of the annotation file.

    Returns:
        list[dict]: List of annotations sorted by timestamps.
    """
    data = mmcv.load(ann_file)
    return data
    # return data_infos

def error_callback(x):
    print(f'error call back {x}')

def single_cpu_merge(proc_id, ann_info,FILE_EXIST_COUNT, AREA_COUNT):
    info = parse_info_bevimage(ann_info, DATA_ROOT, FILE_EXIST_COUNT, AREA_COUNT)
    return None


# 定义count来统计个数
FILE_EXIST_COUNT = 0
AREA_COUNT = 0

# shiyu: main function
ann_info = load_annotations(ann_file)

# 多进程
cpu_num = multiprocessing.cpu_count()
# gt_mask_names = os.listdir(gt_labels_root)
multi_splits = np.array_split(ann_info, cpu_num)
workers = multiprocessing.Pool(processes=cpu_num)
for proc_id, sub_splits in enumerate(multi_splits):
    workers.apply_async(single_cpu_merge, (proc_id, sub_splits, FILE_EXIST_COUNT, AREA_COUNT),
                        error_callback=error_callback)
workers.close()
workers.join()

print("--------------AREA_COUNT is:--------------")
print(AREA_COUNT)
print("--------------FILE_EXIST_COUNT is:--------------")
print(FILE_EXIST_COUNT)