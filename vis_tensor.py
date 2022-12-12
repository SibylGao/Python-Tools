
import pdb; pdb.set_trace()
PATH_vis = "/home/users/gaoshiyu01/dsgn_vis/"
depth_map_stage1_vis = batch_dict['depth_preds_list'][0][0,:].clone().detach().cpu().numpy()
depth_map_stage2_vis = batch_dict['depth_preds_list'][1][0,:].clone().detach().cpu().numpy()
depth_map_local_stage1_vis = batch_dict['depth_preds_local_list'][0][0,:].clone().detach().cpu().numpy()
depth_map_local_stage2_vis = batch_dict['depth_preds_local_list'][1][0,:].clone().detach().cpu().numpy()
gt_vis = batch_dict['depth_gt_img'][0][0,:].clone().detach().cpu().numpy()

mask = (gt_vis > self.min_depth) & (gt_vis < self.max_depth)

dist_stage1 = abs(depth_map_stage1_vis - gt_vis)[mask]
dist_stage2 = abs(depth_map_stage2_vis - gt_vis)[mask]
import pdb; pdb.set_trace()
dist_stage1 = self.img_pro_shiyu(dist_stage1)
dist_stage2 = self.img_pro_shiyu(dist_stage2)
# import pdb; pdb.set_trace()
depth_map_stage1_vis = self.img_pro_shiyu(depth_map_stage1_vis)
depth_map_stage2_vis = self.img_pro_shiyu(depth_map_stage2_vis)
depth_map_local_stage1_vis = self.img_pro_shiyu(depth_map_local_stage1_vis)
depth_map_local_stage2_vis = self.img_pro_shiyu(depth_map_local_stage2_vis)
gt_vis = self.img_pro_shiyu(gt_vis)

cv2.imwrite(PATH_vis + 'depth_map_stage1_vis.jpg', depth_map_stage1_vis)
cv2.imwrite(PATH_vis + 'depth_map_stage2_vis.jpg', depth_map_stage2_vis)
cv2.imwrite(PATH_vis + 'depth_map_local_stage1_vis.jpg', depth_map_local_stage1_vis)
cv2.imwrite(PATH_vis + 'depth_map_local_stage2_vis.jpg', depth_map_local_stage2_vis)
cv2.imwrite(PATH_vis + 'gt_vis.jpg', gt_vis)
cv2.imwrite(PATH_vis + 'dist_stage1.jpg', dist_stage1)
cv2.imwrite(PATH_vis + 'dist_stage2.jpg', dist_stage2)
