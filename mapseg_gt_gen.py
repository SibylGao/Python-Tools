import copy
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
from mmcv.runner import force_fp32, auto_fp16
import torch

# shiyu add: build mask seg head
from ..builder import build_head
import cv2
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt


CANDIDATE=['n008-2018-08-01-15-16-36-0400_1533151184047036',
           'n008-2018-08-01-15-16-36-0400_1533151200646853',
           'n008-2018-08-01-15-16-36-0400_1533151274047332',
           'n008-2018-08-01-15-16-36-0400_1533151369947807',
           'n008-2018-08-01-15-16-36-0400_1533151581047647',
           'n008-2018-08-01-15-16-36-0400_1533151585447531',
           'n008-2018-08-01-15-16-36-0400_1533151741547700',
           'n008-2018-08-01-15-16-36-0400_1533151854947676',
           'n008-2018-08-22-15-53-49-0400_1534968048946931',
           'n008-2018-08-22-15-53-49-0400_1534968255947662',
           'n008-2018-08-01-15-16-36-0400_1533151616447606',
           'n015-2018-07-18-11-41-49+0800_1531885617949602',
           'n008-2018-08-28-16-43-51-0400_1535489136547616',
           'n008-2018-08-28-16-43-51-0400_1535489145446939',
           'n008-2018-08-28-16-43-51-0400_1535489152948944',
           'n008-2018-08-28-16-43-51-0400_1535489299547057',
           'n008-2018-08-28-16-43-51-0400_1535489317946828',
           'n008-2018-09-18-15-12-01-0400_1537298038950431',
           'n008-2018-09-18-15-12-01-0400_1537298047650680',
           'n008-2018-09-18-15-12-01-0400_1537298056450495',
           'n008-2018-09-18-15-12-01-0400_1537298074700410',
           'n008-2018-09-18-15-12-01-0400_1537298088148941',
           'n008-2018-09-18-15-12-01-0400_1537298101700395',
           'n015-2018-11-21-19-21-35+0800_1542799330198603',
           'n015-2018-11-21-19-21-35+0800_1542799345696426',
           'n015-2018-11-21-19-21-35+0800_1542799353697765',
           'n015-2018-11-21-19-21-35+0800_1542799525447813',
           'n015-2018-11-21-19-21-35+0800_1542799676697935',
           'n015-2018-11-21-19-21-35+0800_1542799758948001',
           ]


DEBUG_CANDIDATE=[
    'n008-2018-08-01-15-16-36-0400_1533151212898782',
    'n008-2018-08-01-15-16-36-0400_1533151274047332'
]

# shiyu add
def img_pro_shiyu(img_pro):
    # import pdb; pdb.set_trace()
    img_min = np.min(img_pro)
    img_max = np.max(img_pro)
    interval = img_max - img_min
    new_img = (img_pro - img_min)/interval
    new_img = new_img * 255.0
    return new_img 

def gen_random_color():
    random_RGB = np.random.randint(50,255,size=3)
    # random_r = random.randint(0,255)
    # random_g = random.randint(0,255)
    # random_b = random.randint(0,255)
    # random_RGB = [random_r,random_g,random_b]
    return random_RGB

def cal_iou(gt_mask, pre_mask):
    # import pdb; pdb.set_trace()
    intersect = (gt_mask & pre_mask).astype(np.int)
    # union = int(gt_mask | pre_mask)
    union = (gt_mask).astype(np.int)
    intersection_count = np.sum(intersect)
    union_count = np.sum(union)
    if union_count==0: return 0
    return round((intersection_count/union_count), 2)

# 计算一个类别里面所有的ious
def cal_ious_list(gt_mask_list, pre_mask_list):
    len_gt = len(gt_mask_list)
    len_dt = len(pre_mask_list)
    ious = np.empty([len_dt,len_gt], dtype = float)
    for i in range(len_dt):
        for j in range(len_gt):
            iou = cal_iou(gt_mask_list[j].astype(bool),pre_mask_list[i])
            ious[i][j] = iou
    return ious

# 顺序： front, front_right, front_left, cam_back, cam_back_left, cam_back_right
def draw_pv_img(img_list):
    new_size = [1067,600] # width, height
    front_img = cv2.imread(img_list[0],1)   # flag=1,bgr 3 channels
    front_right = cv2.imread(img_list[1],1)
    front_left = cv2.imread(img_list[2],1)
    cam_back = cv2.imread(img_list[3],1)
    cam_back_left = cv2.imread(img_list[4],1)
    cam_back_right = cv2.imread(img_list[5],1)

    new_front = cv2.resize(front_img, new_size, interpolation=cv2.INTER_LINEAR)
    new_front_right = cv2.resize(front_right, new_size, interpolation=cv2.INTER_LINEAR)
    new_front_left = cv2.resize(front_left, new_size, interpolation=cv2.INTER_LINEAR)
    new_cam_back = cv2.resize(cam_back, new_size, interpolation=cv2.INTER_LINEAR)
    new_cam_back_left = cv2.resize(cam_back_left, new_size, interpolation=cv2.INTER_LINEAR)
    new_cam_back_right = cv2.resize(cam_back_right, new_size, interpolation=cv2.INTER_LINEAR)

    row_1 = np.hstack([new_front_left,new_front,new_front_right])
    row_2 = np.hstack([new_cam_back_left, new_cam_back, new_cam_back_right])
    # import pdb; pdb.set_trace()
    # [1200,3200]
    result = np.vstack([row_1,row_2])

    return result

def draw_pv_img_low_res(img_list):
    # new_size = [1067,600] # width, height
    new_size = [178,100] # width, height
    front_img = cv2.imread(img_list[0],1)   # flag=1,bgr 3 channels
    front_right = cv2.imread(img_list[1],1)
    front_left = cv2.imread(img_list[2],1)
    cam_back = cv2.imread(img_list[3],1)
    cam_back_left = cv2.imread(img_list[4],1)
    cam_back_right = cv2.imread(img_list[5],1)

    new_front = cv2.resize(front_img, new_size, interpolation=cv2.INTER_LINEAR)
    new_front_right = cv2.resize(front_right, new_size, interpolation=cv2.INTER_LINEAR)
    new_front_left = cv2.resize(front_left, new_size, interpolation=cv2.INTER_LINEAR)
    new_cam_back = cv2.resize(cam_back, new_size, interpolation=cv2.INTER_LINEAR)
    new_cam_back_left = cv2.resize(cam_back_left, new_size, interpolation=cv2.INTER_LINEAR)
    new_cam_back_right = cv2.resize(cam_back_right, new_size, interpolation=cv2.INTER_LINEAR)

    row_1 = np.hstack([new_front_left,new_front,new_front_right])
    row_2 = np.hstack([new_cam_back_left, new_cam_back, new_cam_back_right])
    # import pdb; pdb.set_trace()
    # [1200,3200]
    result = np.vstack([row_1,row_2])

    return result

# 画出 ins 匹配之后的实例
def draw_ins_w_match(gts, dts, gt_match, dt_match, gt_label, bev_feat_list):

    bev_feat = bev_feat_list[0][0,:,:,:]    # c,h,w
    bev_feat = torch.mean(bev_feat,0,keepdim=True)  #1,h,w
    bev_feat = bev_feat.permute(1,2,0)
    bev_feat_vis = bev_feat.clone().detach().cpu().numpy()
    dt_img = cv2.cvtColor(bev_feat_vis, cv2.COLOR_GRAY2BGR) #h,w,3

    dt_img = np.ones([200,100,3])*255

    # new_mask_size = [600,1200]

    # import pdb; pdb.set_trace()
    gt_label = gt_label.clone().detach().cpu().numpy()
    for i in range(len(gt_match)):
        single_dt_idx = gt_match[i]         # 和dt match的参数
        if single_dt_idx==(-1): continue
        color = LABEL2COLOR[gt_label[i]]
        single_dt = dts[single_dt_idx]
        dt_img[single_dt] = color
    return dt_img


# pre_masks 已经排序好了
def match_masks(gt_masks, pre_masks, thr=0.0):
    height = len(pre_masks)
    width = len(gt_masks)
    # gt_masks = gt_masks.astype(bool)
    pre_masks = pre_masks.astype(bool)
    ious = cal_ious_list(gt_masks, pre_masks)
    gt_match = np.empty([width], dtype = int)
    dt_match = np.empty([height], dtype = int)
    gt_match.fill(-1)
    dt_match.fill(-1)
    for i in range(height):  # dt
        single_ious = ious[i]
        if len(single_ious)==0: continue
        max_idx = np.argmax(single_ious)
        # max_idx = single_ious.index(max(single_ious))
        max_iou = ious[i][max_idx]
        if gt_match[max_idx] > 0: dt_match[i] = -2; continue  # 如果已经匹配了，就继续运行
        if max_iou<thr: continue
        dt_match[i] = max_idx
        gt_match[max_idx] = i
    return dt_match, gt_match,ious


LABEL2COLOR={
    0: [255,0,0],
    1: [0,255,0],
    2: [0,0,255]
}
# feat_save_root = '/mnt/vepfs/ML/ml-users/shiyu/MapTR_local/mapseg_vis_highres/'
def draw_gt_bev_on_feature(gt_bboxes_3d, gt_label_list, img_metas, bev_feat_list,feat_save_root):
    # import pdb; pdb.set_trace()
    img_name = img_metas[0]['scene_token'] + '.png'
    bev_feat = bev_feat_list[0][0,:,:,:]    # c,h,w
    bev_feat = torch.mean(bev_feat,0,keepdim=True)  #1,h,w
    bev_feat = bev_feat.permute(1,2,0)
    bev_feat_vis = bev_feat.clone().detach().cpu().numpy()
    bev_feat_vis_color = cv2.cvtColor(bev_feat_vis, cv2.COLOR_GRAY2BGR) #h,w,3

    raw_img = bev_feat.clone().detach().cpu().numpy()
    raw_img = img_pro_shiyu(raw_img)
    raw_img = cv2.cvtColor(raw_img, cv2.COLOR_GRAY2BGR) #h,w,3
    bev_mask = gt_bboxes_3d[0].masks_list   # num_mask, h, w
    gt_label = gt_label_list[0]
    gt_label_cpu = gt_label.clone().detach().cpu().numpy()
    for i in range(len(bev_mask)):
        single_instance_mask = (bev_mask[i]).astype(bool)    #h,w
        # single_instance_mask = np.expand_dims(single_instance_mask,2).repeat(3,axis=2)
        label = gt_label_cpu[i]
        COLOR = LABEL2COLOR[label]
        bev_feat_vis_color[single_instance_mask] = COLOR
    result = np.hstack([raw_img,bev_feat_vis_color])
    feat_save_path = feat_save_root + img_name
    raw_save_path = feat_save_root + 'raw_img' + img_name
    cv2.imwrite(feat_save_path, result)
    # cv2.imwrite(raw_save_path, raw_img)


def draw_gt_img(mask_list, gt_label_list):
    # bg 是白色
    gt_img = np.ones([200,100,3]) * 255
    gt_labels = gt_label_list.clone().detach().cpu().numpy()
    # import pdb; pdb.set_trace()
    for i in range(len(mask_list)):
        label = gt_labels[i]
        COLOR = LABEL2COLOR[label]
        single_instance_mask = mask_list[i].clone().detach().cpu().numpy().astype(bool)
        gt_img[single_instance_mask] = COLOR
    return gt_img

def draw_gt_bev_on_feature_w_pred_low_res(gt_bboxes_3d, gt_label_list, pts_filename, bev_feat_list, pred_img, feat_save_root, pv_img=None):
    gt_new_size = [600,1200]
    Car_COLOR = [255,255,0]
    Car_position = [100,50]
    # Car_position = [600,300]
    car = cv2.imread('figs/car.png',1)
    car_resize = cv2.resize(car, [8,10], interpolation=cv2.INTER_LINEAR)
    car_img = np.zeros([1200,600,3])
    # x,y = 300,600
    x,y = 50,100
    # import pdb; pdb.set_trace()
    car_resize = cv2.flip(cv2.transpose(car_resize), 1)
    car_img[y:y + 8, x:x + 10,:] = car_resize

    img_name = pts_filename + '.png'
    bev_feat = bev_feat_list[0][0,:,:,:]    # c,h,w
    bev_feat = torch.mean(bev_feat,0,keepdim=True)  #1,h,w
    bev_feat = bev_feat.permute(1,2,0)
    bev_feat_vis = bev_feat.clone().detach().cpu().numpy()
    bev_feat_vis_color = cv2.cvtColor(bev_feat_vis, cv2.COLOR_GRAY2BGR) #h,w,3

    gt_img = np.ones([200,100,3])*255

    raw_img = bev_feat.clone().detach().cpu().numpy()
    raw_img = img_pro_shiyu(raw_img)
    raw_img = cv2.cvtColor(raw_img, cv2.COLOR_GRAY2BGR) #h,w,3
    bev_mask = gt_bboxes_3d[0].masks_list   # num_mask, h, w
    gt_label = gt_label_list[0]
    gt_label_cpu = gt_label.clone().detach().cpu().numpy()
    for i in range(len(bev_mask)):
        single_instance_mask = (bev_mask[i]).astype(bool)    #h,w
        # single_instance_mask = np.expand_dims(single_instance_mask,2).repeat(3,axis=2)
        label = gt_label_cpu[i]
        COLOR = LABEL2COLOR[label]
        gt_img[single_instance_mask] = COLOR
    
    # resize to higher resolution
    # gt_img = cv2.resize(gt_img, gt_new_size, interpolation=cv2.INTER_LINEAR)
    # pred_img = cv2.resize(pred_img, gt_new_size, interpolation=cv2.INTER_LINEAR)

    gt_img[y:y + 8, x:x + 10,:] = car_resize
    pred_img[y:y + 8, x:x + 10,:] = car_resize

    seg = np.zeros([200,3,3])
    # seg = np.zeros([1200,3,3])
    # seg = seg*255
    # import pdb; pdb.set_trace()
    result = np.hstack([gt_img,seg,pred_img])
    # result = np.hstack([raw_img,gt_img,seg,pred_img])
    feat_save_path = feat_save_root + 'low_res' + img_name
    raw_save_path = feat_save_root + 'raw_img' + img_name

    # import pdb; pdb.set_trace()
    result = np.hstack([pv_img, result])
    cv2.imwrite(feat_save_path, result)

def draw_gt_bev_on_feature_w_pred(gt_bboxes_3d, gt_label_list, pts_filename, bev_feat_list, pred_img, feat_save_root, pv_img=None):
    # import pdb; pdb.set_trace()
    # pv_img # [1200,3200]
    gt_new_size = [600,1200]
    Car_COLOR = [255,255,0]
    # Car_position = [100,50]
    Car_position = [600,300]
    car = cv2.imread('figs/car.png',1)
    car_resize = cv2.resize(car, [16,20], interpolation=cv2.INTER_LINEAR)
    car_img = np.zeros([1200,600,3])
    x,y = 300,600
    # import pdb; pdb.set_trace()
    car_resize = cv2.flip(cv2.transpose(car_resize), 1)
    car_img[y:y + 16, x:x + 20,:] = car_resize

    img_name = pts_filename + '.png'
    bev_feat = bev_feat_list[0][0,:,:,:]    # c,h,w
    bev_feat = torch.mean(bev_feat,0,keepdim=True)  #1,h,w
    bev_feat = bev_feat.permute(1,2,0)
    bev_feat_vis = bev_feat.clone().detach().cpu().numpy()
    bev_feat_vis_color = cv2.cvtColor(bev_feat_vis, cv2.COLOR_GRAY2BGR) #h,w,3

    gt_img = np.ones([200,100,3])*255

    raw_img = bev_feat.clone().detach().cpu().numpy()
    raw_img = img_pro_shiyu(raw_img)
    raw_img = cv2.cvtColor(raw_img, cv2.COLOR_GRAY2BGR) #h,w,3
    bev_mask = gt_bboxes_3d[0].masks_list   # num_mask, h, w
    gt_label = gt_label_list[0]
    gt_label_cpu = gt_label.clone().detach().cpu().numpy()
    for i in range(len(bev_mask)):
        single_instance_mask = (bev_mask[i]).astype(bool)    #h,w
        # single_instance_mask = np.expand_dims(single_instance_mask,2).repeat(3,axis=2)
        label = gt_label_cpu[i]
        COLOR = LABEL2COLOR[label]
        gt_img[single_instance_mask] = COLOR
    
    # resize to higher resolution
    gt_img = cv2.resize(gt_img, gt_new_size, interpolation=cv2.INTER_LINEAR)
    pred_img = cv2.resize(pred_img, gt_new_size, interpolation=cv2.INTER_LINEAR)

    # draw car
    # cv2.rectangle(bev_feat_vis_color, (99, 49), (101, 51), (0,0,255), 2)
    # cv2.rectangle(pred_img, (99, 49), (101, 51), (0,0,255), 2)
    # gt_img = gt_img + car_img
    # pred_img = pred_img + car_img
    # import pdb; pdb.set_trace()
    gt_img[y:y + 16, x:x + 20,:] = car_resize
    pred_img[y:y + 16, x:x + 20,:] = car_resize

    seg = np.zeros([1200,3,3])
    # seg = seg*255
    # import pdb; pdb.set_trace()
    result = np.hstack([gt_img,seg,pred_img])
    # result = np.hstack([raw_img,gt_img,seg,pred_img])
    feat_save_path = feat_save_root + img_name
    raw_save_path = feat_save_root + 'raw_img' + img_name

    # import pdb; pdb.set_trace()
    result = np.hstack([pv_img, result])
    cv2.imwrite(feat_save_path, result)
    # cv2.imwrite(raw_save_path, raw_img)


def draw_pred_bev_on_feature(pred_masks_list, gt_label_list, img_metas, bev_feat_list, feat_save_root):
    # import pdb; pdb.set_trace()
    img_name = img_metas[0]['scene_token'] + '.png'
    bev_feat = bev_feat_list[0][0,:,:,:]    # c,h,w
    bev_feat = torch.mean(bev_feat,0,keepdim=True)  #1,h,w
    bev_feat = bev_feat.permute(1,2,0)
    bev_feat_vis = bev_feat.clone().detach().cpu().numpy()
    bev_feat_vis_color = cv2.cvtColor(bev_feat_vis, cv2.COLOR_GRAY2BGR) #h,w,3

    raw_img = bev_feat.clone().detach().cpu().numpy()
    raw_img = img_pro_shiyu(raw_img)
    raw_img = cv2.cvtColor(raw_img, cv2.COLOR_GRAY2BGR) #h,w,3
    # 取最后一层
    bev_mask = pred_masks_list[-1][0]  # num_query, h, w
    gt_label = gt_label_list[0]
    gt_label_cpu = gt_label.clone().detach().cpu().numpy()
    for i in range(len(bev_mask)):
        single_instance_mask = (bev_mask[i]).astype(bool)    #h,w
        # single_instance_mask = np.expand_dims(single_instance_mask,2).repeat(3,axis=2)
        label = gt_label_cpu[i]
        COLOR = LABEL2COLOR[label]
        bev_feat_vis_color[single_instance_mask] = COLOR
    result = np.hstack([raw_img,bev_feat_vis_color])
    feat_save_path = feat_save_root + 'pred_mask' + img_name
    raw_save_path = feat_save_root + 'raw_img' + img_name
    cv2.imwrite(feat_save_path, result)
    # cv2.imwrite(raw_save_path, raw_img)

# curb=0,0,255
def draw_single_mask(single_mask, COLOR=[0,0,255]):
    # img_save_path = root + img_name
    vis_img = np.ones([200,100,3]) *255
    single_mask_bool = single_mask.clone().detach().cpu().numpy().astype(bool)

    vis_img[single_mask_bool] = COLOR

    return vis_img, single_mask_bool

def get_nerighbor_label(label, rigion_mask, rigion_label):
    '''
        label: label map
        rigion_mask: bool mask for single connected rigion
    '''
    kernel = np.ones((3, 3), np.uint8)
    dila_rigion = cv2.dilate(rigion_mask,kernel)
    neighbor_rigion = label[dila_rigion.astype(bool)]
    # import pdb; pdb.set_trace()
    neighbor_label = np.unique(neighbor_rigion)
    neighbor_label.tolist().remove(rigion_label)
    return neighbor_label

@DETECTORS.register_module()
class MapTRSeg(MVXTwoStageDetector):
    """MapTR.
    Args:
        video_test_mode (bool): Decide whether to use temporal information during inference.
    """

    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 # shiyu add
                 mask_seg_head=None,
                 seg_train_cfg=None,
                 seg_test_cfg=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 video_test_mode=False
                 ):

        super(MapTRSeg,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False

        # temporal
        self.video_test_mode = video_test_mode
        self.prev_frame_info = {
            'prev_bev': None,
            'scene_token': None,
            'prev_pos': 0,
            'prev_angle': 0,
        }


        # shiyu add mask_seg_head initialize
        panoptic_head_ = copy.deepcopy(mask_seg_head)
        panoptic_head_.update(train_cfg=seg_train_cfg)
        panoptic_head_.update(test_cfg=seg_test_cfg)
        # import pdb; pdb.set_trace()
        self.panoptic_head = build_head(panoptic_head_)

        self.num_things_classes = self.panoptic_head.num_things_classes
        self.num_stuff_classes = self.panoptic_head.num_stuff_classes
        self.num_classes = self.panoptic_head.num_classes

    def extract_img_feat(self, img, img_metas, len_queue=None):
        """Extract features of images."""
        B = img.size(0)
        if img is not None:
            
            # input_shape = img.shape[-2:]
            # # update real input shape of each single img
            # for img_meta in img_metas:
            #     img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)

            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(int(B/len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_feat(self, img, img_metas=None, len_queue=None):
        """Extract features from images and points."""

        img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)
        
        return img_feats


    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None,
                          prev_bev=None):
        """Forward function'
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
            prev_bev (torch.Tensor, optional): BEV features of previous frame.
        Returns:
            dict: Losses of each branch.
        """
        # shiyu add: return bev feat list only
        bev_feat_list = self.pts_bbox_head(
            pts_feats, img_metas, prev_bev, only_bev=True)
        
        # vis feat
        # draw_gt_bev_on_feature(gt_bboxes_3d, gt_labels_3d, img_metas, bev_feat_list)


        # import pdb; pdb.set_trace()
        # shiyu add panoptic_head forward
        # list of [b, num_query, class], [b, num_query, h, w]
        all_cls_scores, all_mask_preds = self.panoptic_head(bev_feat_list, img_metas)

        # check candidate, if in candidate visualize
        pts_filename = img_metas[0]['pts_filename']
        pts_filename = osp.basename(pts_filename)
        pts_filename = pts_filename.replace('__LIDAR_TOP__', '_').split('.')[0]
        # if pts_filename in CANDIDATE:
            # shiyu add: match && visualize
        mask_pred = all_mask_preds[-1][0]
        mask_pred = mask_pred.sigmoid()
        mask_bool = mask_pred > 0.5     # mask bool

        mask_pred_cpu_bool = mask_bool.clone().detach().cpu().numpy().astype(bool)   # num_query, h, w
        # mask_pred_cpu = all_mask_preds[-1][0].clone().detach().cpu().numpy()   # num_query, h, w
        dt_match, gt_match, ious = match_masks(gt_bboxes_3d[0].masks_list, mask_pred_cpu_bool)

        # import pdb; pdb.set_trace()
        pred_mask = draw_ins_w_match(gt_bboxes_3d[0].masks_list, mask_pred_cpu_bool, gt_match, dt_match, gt_labels_3d[0], bev_feat_list)
        pv_img = draw_pv_img(img_metas[0]['filename'])
        pv_img_low_res = draw_pv_img_low_res(img_metas[0]['filename'])
        if pts_filename in CANDIDATE:
            feat_save_root = "/mnt/vepfs/ML/ml-users/shiyu/MapTR_local/mapseg_select_vis_epoch110/"
        else:
            feat_save_root = "/mnt/vepfs/ML/ml-users/shiyu/MapTR_local/mapseg_vis_highres/"
        
        # 画图
        # draw_gt_bev_on_feature_w_pred(gt_bboxes_3d, gt_labels_3d, pts_filename, bev_feat_list, pred_mask, feat_save_root, pv_img=pv_img)
        # draw_gt_bev_on_feature_w_pred_low_res(gt_bboxes_3d, gt_labels_3d, pts_filename, bev_feat_list, pred_mask, pv_img=pv_img_low_res)
        
        # import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()
        # preprocess ground truth
        # gt_labels, gt_masks = self.panoptic_head.preprocess_gt(gt_labels, gt_masks,
        #                                          gt_semantic_seg, img_metas)
        # import pdb; pdb.set_trace()
        # draw_pred_bev_on_feature(all_mask_preds, gt_labels_3d, img_metas, bev_feat_list)

        # shiyu add vis single mask


        # 将mask list从gt中分离出来
        gt_mask_lists = []
        for gt_bbox in gt_bboxes_3d:
            gt_masks = gt_bbox.masks_list
            single_frame_mask_list = []
            for i in range(len(gt_masks)):
                single_mask = gt_masks[i]
                single_mask_tensor = torch.from_numpy(single_mask).to(gt_labels_3d[0].device)
                single_frame_mask_list.append(single_mask_tensor)
            single_frame_mask = torch.stack(single_frame_mask_list,0)
            gt_mask_lists.append(single_frame_mask)
        
        # shiyu add
        single_mask_test_root = "/mnt/vepfs/ML/ml-users/shiyu/MapTR_local/single_mask_test_root5/"
        gt_mask_vis_single = draw_gt_img(gt_mask_lists[0],gt_labels_3d[0]) # 200,100,3
        outer_mask_img = np.ones([200,100,3]) * 255
        mask_curbs = np.ones([200,100]) 
        img_curb = np.ones([200,100,3]) * 255
        # mask_lanes_n_cross = np.ones([200,100,3])
        # curb_lines_list = []
        mask_root = single_mask_test_root + pts_filename + '_mask_curb' + ".png"
        for i in range(len(gt_labels_3d[0])):
            # 如果这个ins是curb的话
            if gt_labels_3d[0][i]==2:
                mask_img, mask_bool = draw_single_mask(gt_mask_lists[0][i])
                mask_path = single_mask_test_root + pts_filename + '_mask_curb_{}_'.format(i) + ".png"
                seg = np.zeros([200,3,3])
                mask_curbs[mask_bool] = 0
                
                # mask_curbs[mask_bool] = 1
        # # 连通域求错了
        # img_curb = np.sum(img_curb,2)   # single channel
        # mask_gray = img_curb > 255      # 有label的地方是0
        # mask_gray = ~mask_curbs     # 有label的地方是0
        mask_curb_root = single_mask_test_root + 'curb_mask.png'
        # cv2.imwrite(mask_curb_root, mask_curbs.astype(np.uint8)*255)
        # 连通域需要轮廓是黑色，值为0，
        # 但是膨胀腐蚀需要轮廓是白色，值为1
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_curbs.astype(np.uint8)*255, connectivity=4)
        labels_root = single_mask_test_root + 'labels.png'
        # cv2.imwrite(labels_root, labels.astype(np.uint8)*30)
        # import pdb; pdb.set_trace()
        # 找到curb线所在的label编号
        line_list = []
        for i in range(num_labels):
            single_area = (labels==i)
            union = ~(mask_curbs.astype(bool)) & single_area
            if union.any(): line_list.append(i)


        drivable_area_label = []
        curb_area_label = []

        center_label = labels[100,50]
        center_rigion = (labels==center_label)
        drivable_area_label.append(center_label)

        tmp_list = []
        tmp_list.append(center_label)
        visited_list = []

        while tmp_list:
            current_label = tmp_list.pop()
            if current_label in visited_list: continue
            visited_list.append(current_label)
            if current_label in line_list: continue
            current_area = (labels==current_label)
            tmp_root = single_mask_test_root + 'tmp.png'
            # import pdb; pdb.set_trace()
            # img_curb[current_area] = [0,0,255]
            # cv2.imwrite(tmp_root, img_curb)
            # 邻域的所有标签
            neighbor_label = get_nerighbor_label(labels, current_area.astype(np.uint8), current_label)
            # 相邻连通域是否可行驶是相反的
            if current_label in drivable_area_label: 
                neighbor_label_filter = set(neighbor_label).difference(set(drivable_area_label))
                curb_area_label=list(set(curb_area_label).union(neighbor_label_filter))
            elif current_label in curb_area_label: 
                neighbor_label_filter = set(neighbor_label).difference(set(curb_area_label))
                drivable_area_label=list(set(drivable_area_label).union(neighbor_label_filter))
            else: print("current label not in any area!")

            tmp_list = list(set(tmp_list).union(set(neighbor_label)))
            # import pdb; pdb.set_trace()
            # print(tmp_list)
            # print("---------------visited list--------------")
            # print(visited_list)
            

        for i in range(len(curb_area_label)):
            single_curb_label = curb_area_label[i]
            single_curb_mask = (labels==single_curb_label)
            outer_mask_img[single_curb_mask] = [0,0,255]

        for i in range(len(line_list)):
            single_curb_label = line_list[i]
            single_curb_mask = (labels==single_curb_label)
            outer_mask_img[single_curb_mask] = [0,0,255]

        connect_label = cv2.cvtColor(labels.astype(np.float32),cv2.COLOR_GRAY2BGR)*30
        result = np.hstack([gt_mask_vis_single,connect_label,outer_mask_img])
    
        # 画ped crossing

        cv2.imwrite(mask_root, result)
        import pdb; pdb.set_trace()
        
        # # if pts_filename in DEBUG_CANDIDATE: import pdb; pdb.set_trace()
        # # shiyu add vis
        # gt_mask_vis_single = draw_gt_img(gt_mask_lists[0],gt_labels_3d[0]) # 200,100,3
        # single_mask_test_root = "/mnt/vepfs/ML/ml-users/shiyu/MapTR_local/single_mask_test_root3/"
        # mask_root = single_mask_test_root + pts_filename + '_mask_curb' + ".png"
        # outer_mask_img = np.ones([200,100,3]) * 255
        # seg = np.zeros([200,3,3])
        # # mask_curbs = np.ones([200,100,3]) * 255
        # for i in range(len(gt_labels_3d[0])):
        #     # 对于一个single instance
        #     if gt_labels_3d[0][i]==2:
        #         mask_img, single_mask_bool = draw_single_mask(gt_mask_lists[0][i])
        #         mask_path = single_mask_test_root + pts_filename + '_mask_curb_{}_'.format(i) + ".png"
        #         seg = np.zeros([200,3,3])

        #         # result_mask = np.hstack([gt_mask_vis_single,seg])
        #         result_mask = np.hstack([gt_mask_vis_single,seg,mask_img])

        #         # 画连通域
        #         # import pdb; pdb.set_trace()
        #         mask_img = np.sum(mask_img,2)   # single channel
        #         mask_gray = mask_img > 255      # 有label的地方是0
        #         # mask_gray = cv2.cvtColor(mask_img.astype(np.float32)/255,cv2.COLOR_BGR2GRAY)
        #         num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_gray.astype(np.uint8), connectivity=4)
        #         # import pdb; pdb.set_trace()
        #         # bev_center = [100,50]
        #         center_label = labels[100,50]
        #         unique_label = np.unique(labels)
        #         num_labels = len(unique_label)
        #         areas = stats[:,4]
        #         line_ind = np.argmin(areas)      # min area is line
                
        #         outer_label=0
        #         max_area = 0
        #         for i in range(num_labels):
        #             # if i == line_ind: continue
        #             single_label = unique_label[i]
        #             if single_label == center_label: continue
        #             single_area = areas[i]
        #             if single_area>max_area:
        #                 max_area = single_area
        #                 outer_label = unique_label[i]
                
        #         outer_mask = labels==outer_label    #取这个mask
        #         # outer_mask_img = np.ones([200,100,3]) * 255
        #         outer_mask_img[outer_mask] = [0,0,255]

        #         # import pdb; pdb.set_trace()
        #         connect_label = cv2.cvtColor(labels.astype(np.float32),cv2.COLOR_GRAY2BGR)*100

                
        #         result_mask = np.hstack([result_mask, connect_label, outer_mask_img])

        #         # cv2.imwrite(mask_root, np.hstack([gt_mask_vis_single, seg, outer_mask_img]))
        #         if pts_filename in DEBUG_CANDIDATE: 
        #             cv2.imwrite(mask_path, result_mask)
        #             import pdb; pdb.set_trace()
        #     # import pdb; pdb.set_trace()

        # import pdb;pdb.set_trace()

        # # loss
        losses = self.panoptic_head.loss(all_cls_scores, all_mask_preds, gt_labels_3d, gt_mask_lists,
                           img_metas)
        

        # 梯度清零，防止模型更新
        # import pdb; pdb.set_trace()
        new_losses = dict()
        for key,value in losses.items():
            new_losses[key] = value * 0.0
        
        # import pdb; pdb.set_trace
        return new_losses
        # return losses

        # loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        # losses = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)
        # return losses

    def forward_dummy(self, img):
        dummy_metas = None
        return self.forward_test(img=img, img_metas=[[dummy_metas]])

    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)
    
    def obtain_history_bev(self, imgs_queue, img_metas_list):
        """Obtain history BEV features iteratively. To save GPU memory, gradients are not calculated.
        """
        self.eval()

        with torch.no_grad():
            prev_bev = None
            bs, len_queue, num_cams, C, H, W = imgs_queue.shape
            imgs_queue = imgs_queue.reshape(bs*len_queue, num_cams, C, H, W)
            img_feats_list = self.extract_feat(img=imgs_queue, len_queue=len_queue)
            for i in range(len_queue):
                img_metas = [each[i] for each in img_metas_list]
                # img_feats = self.extract_feat(img=img, img_metas=img_metas)
                img_feats = [each_scale[:, i] for each_scale in img_feats_list]
                prev_bev = self.pts_bbox_head(
                    img_feats, img_metas, prev_bev, only_bev=True)
            self.train()
            return prev_bev

    # @auto_fp16(apply_to=('img', 'points'))
    @force_fp32(apply_to=('img','points','prev_bev'))
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      img_depth=None,
                      img_mask=None,
                      ):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        
        len_queue = img.size(1)
        prev_img = img[:, :-1, ...]
        img = img[:, -1, ...]

        prev_img_metas = copy.deepcopy(img_metas)
        # prev_bev = self.obtain_history_bev(prev_img, prev_img_metas)
        # import pdb;pdb.set_trace()
        prev_bev = self.obtain_history_bev(prev_img, prev_img_metas) if len_queue>1 else None

        img_metas = [each[len_queue-1] for each in img_metas]
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        losses = dict()
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore, prev_bev)

        losses.update(losses_pts)
        return losses

    def forward_test(self, img_metas, img=None, **kwargs):
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img

        if img_metas[0][0]['scene_token'] != self.prev_frame_info['scene_token']:
            # the first sample of each scene is truncated
            self.prev_frame_info['prev_bev'] = None
        # update idx
        self.prev_frame_info['scene_token'] = img_metas[0][0]['scene_token']

        # do not use temporal information
        if not self.video_test_mode:
            self.prev_frame_info['prev_bev'] = None

        # Get the delta of ego position and angle between two timestamps.
        tmp_pos = copy.deepcopy(img_metas[0][0]['can_bus'][:3])
        tmp_angle = copy.deepcopy(img_metas[0][0]['can_bus'][-1])
        if self.prev_frame_info['prev_bev'] is not None:
            img_metas[0][0]['can_bus'][:3] -= self.prev_frame_info['prev_pos']
            img_metas[0][0]['can_bus'][-1] -= self.prev_frame_info['prev_angle']
        else:
            img_metas[0][0]['can_bus'][-1] = 0
            img_metas[0][0]['can_bus'][:3] = 0

        new_prev_bev, bbox_results = self.simple_test(
            img_metas[0], img[0], prev_bev=self.prev_frame_info['prev_bev'], **kwargs)
        # During inference, we save the BEV features and ego motion of each timestamp.
        self.prev_frame_info['prev_pos'] = tmp_pos
        self.prev_frame_info['prev_angle'] = tmp_angle
        self.prev_frame_info['prev_bev'] = new_prev_bev
        return bbox_results

    def pred2result(self, bboxes, scores, labels, pts, attrs=None):
        """Convert detection results to a list of numpy arrays.

        Args:
            bboxes (torch.Tensor): Bounding boxes with shape of (n, 5).
            labels (torch.Tensor): Labels with shape of (n, ).
            scores (torch.Tensor): Scores with shape of (n, ).
            attrs (torch.Tensor, optional): Attributes with shape of (n, ). \
                Defaults to None.

        Returns:
            dict[str, torch.Tensor]: Bounding box results in cpu mode.

                - boxes_3d (torch.Tensor): 3D boxes.
                - scores (torch.Tensor): Prediction scores.
                - labels_3d (torch.Tensor): Box labels.
                - attrs_3d (torch.Tensor, optional): Box attributes.
        """
        result_dict = dict(
            boxes_3d=bboxes.to('cpu'),
            scores_3d=scores.cpu(),
            labels_3d=labels.cpu(),
            pts_3d=pts.to('cpu'))

        if attrs is not None:
            result_dict['attrs_3d'] = attrs.cpu()

        return result_dict
    

    # def simple_test_pts(self, x, img_metas, prev_bev=None, rescale=False):
    #     """Test function"""
    #     outs = self.pts_bbox_head(x, img_metas, prev_bev=prev_bev)

    #     bbox_list = self.pts_bbox_head.get_bboxes(
    #         outs, img_metas, rescale=rescale)
        
    #     bbox_results = [
    #         self.pred2result(bboxes, scores, labels, pts)
    #         for bboxes, scores, labels, pts in bbox_list
    #     ]
    #     # import pdb;pdb.set_trace()
    #     return outs['bev_embed'], bbox_results
    # def simple_test(self, img_metas, img=None, prev_bev=None, rescale=False, **kwargs):
    #     """Test function without augmentaiton."""
    #     img_feats = self.extract_feat(img=img, img_metas=img_metas)

    #     bbox_list = [dict() for i in range(len(img_metas))]
    #     new_prev_bev, bbox_pts = self.simple_test_pts(
    #         img_feats, img_metas, prev_bev, rescale=rescale)
    #     for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
    #         result_dict['pts_bbox'] = pts_bbox
    #     return new_prev_bev, bbox_list


    # shiyu modified
    def simple_test(self, imgs, img_metas, prev_bev=None, **kwargs):
        feats = self.extract_feat(imgs)

        bev_feat_list = self.pts_bbox_head(
            feats, img_metas, prev_bev, only_bev=True)

        mask_cls_results, mask_pred_results = self.panoptic_head.simple_test(
            bev_feat_list, img_metas, **kwargs)
        
        results = {}
        results['mask_pred'] = mask_pred_results
        results['cls_pred'] = mask_cls_results

        return None, results


@DETECTORS.register_module()
class MapTRSeg_fp16(MapTRSeg):
    """
    The default version BEVFormer currently can not support FP16. 
    We provide this version to resolve this issue.
    """
    # @auto_fp16(apply_to=('img', 'prev_bev', 'points'))
    @force_fp32(apply_to=('img','points','prev_bev'))
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      img_depth=None,
                      img_mask=None,
                      prev_bev=None,
                      ):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        # import pdb;pdb.set_trace()
        losses = dict()
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore, prev_bev=prev_bev)
        losses.update(losses_pts)
        return losses


    def val_step(self, data, optimizer):
        """
        In BEVFormer_fp16, we use this `val_step` function to inference the `prev_pev`.
        This is not the standard function of `val_step`.
        """

        img = data['img']
        img_metas = data['img_metas']
        img_feats = self.extract_feat(img=img,  img_metas=img_metas)
        prev_bev = data.get('prev_bev', None)
        prev_bev = self.pts_bbox_head(img_feats, img_metas, prev_bev=prev_bev, only_bev=True)
        return prev_bev
