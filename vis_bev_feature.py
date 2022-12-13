# shiyu modified
import cv2
import numpy as np
import os

def img_pro_shiyu(self, img_pro):
# import pdb; pdb.set_trace()
img_min = np.min(img_pro)
img_max = np.max(img_pro)
interval = img_max - img_min
new_img = (img_pro - img_min)/interval
new_img = new_img * 255.0
return new_img 

# shiyu modifed vis
# import pdb; pdb.set_trace()
PATH_vis = "/root/paddlejob/workspace/env_run/feature_vis/"
feature_vis = feature[0,0,:,:].clone().detach().cpu().numpy()    #bchw
fea_vis_pro = self.img_pro_shiyu(feature_vis)
fea_vis_pro = fea_vis_pro.astype('uint8')
filename_vis = PATH_vis + "rgb/" + "lidar_feature" + "{}-{}.png".format(metas[0]["timestamp"], metas[0]["token"]) 
cv2.imwrite(filename_vis, fea_vis_pro)
