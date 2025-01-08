import cv2
import torch
import torch.nn.functional as F
import numpy as np

from scipy.special import softmax
from ..utils import maybe_cuda
"""
Tracklet
1) use multi threads to accelerate single->batch and batch->single procedure (for deep-learning-based feature extractor)
"""
class PartTracklet:
    def __init__(self, params, img, observation):
        """
        image (numpy.array): original image with (H,W,3)
        bbox (list(int)): [tl_x, tl_y, br_x, br_y]
        """
        ### parameters ###
        self.params = params
        self.vis_map_size = params.vis_map_size
        self.vis_map_nums = self.vis_map_size[0]
        self.vis_map_res = self.vis_map_size[1:]
        self.part_nums = self.vis_map_nums+1 if not params.use_ori else 2*(self.vis_map_nums+1)

        ### prior parameters ### 
        self.img_scale = (256, 128)
        self.conf_thr = 0.3
        self.temp = 0.1

        ### information ###
        self.bbox = observation[:4]  # Tensor
        self.image_patch = self.crop_imgs(img=img, bbox=self.bbox) # Tensor

        self.bbox_score = observation[4]
        self.kpts = observation[5]
        # [Nose, LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist, LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck]
        self.kpts = np.concatenate((self.kpts, np.expand_dims((np.array(self.kpts)[1, :] + np.array(self.kpts)[2, :]) / 2, 0)), axis=0)
        # if self.params.use_ori:
        self.ori = observation[-1]
        self.binary_ori = self.get_ori(observation[-1])
        self.visibility_indicator, self.visibility_map = self.get_vis_part(self.bbox, self.kpts, observation[-1])  # (4,4,8)

        ### feature ###
        self.deep_feature = None  # (5,512)
        self.att_map = None  # (H,W)-(8,4)
        self.target_confidence = None
        self.part_target_confidence = [None for _ in range(len(self.visibility_indicator))]
    
    def get_vis_part(self, bbox, kpts, ori):
        """Generate visible parts of person
        Input:
            bbox: [tl_x, tl_y, br_x, br_y]
            kpts: [Nose, LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist, LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck]
            ori(degree): orientation that 0-front, 180-back
        Output:
            visible parts(4,4,8;Bool Map): with [head, torso, legs, feet], W, H
        """
        # part_ids = {"head": [0], "torso": [1,2,13], "legs": [7,8,9,10], "feet":[11,12]}
        tl_x, tl_y, br_x, br_y = bbox
        visible_indicator = torch.zeros(self.part_nums)  # front-parts, back-parts
        visible_part_indicator = torch.zeros(self.vis_map_nums)  # parts
        visible_part_map = torch.zeros(self.vis_map_size, dtype=torch.int32)
        parts_ids = [[0], [1,2,13], [7,8,9,10], [11,12]]  # 4 parts
        kpts = torch.Tensor(kpts)
        kpts[:, 0] = torch.clamp(kpts[:, 0], min=tl_x, max=br_x)
        kpts[:, 1] = torch.clamp(kpts[:, 1], min=tl_y, max=br_y)
        kpts = np.array(kpts)
        # print("\nkpts: ", kpts)
        for index, part_ids in enumerate(parts_ids):
            # if all confidences of kpts are smaller than threshold, continue
            if all(kpts[i, 2] < self.conf_thr for i in part_ids):
                continue
            # head
            if index == 0:
                min_part_y = tl_y   
            else:
                part_kpts = kpts[part_ids][kpts[part_ids, 2]>self.conf_thr]
                min_part_y = np.amin(part_kpts[:, 1])
            # feet
            if index == len(parts_ids)-1:
                next_max_part_y = br_y
            else:
                if all(kpts[i, 2] < self.conf_thr for i in parts_ids[index+1]):
                    next_max_part_y = br_y
                else:
                    next_part_kpts = kpts[parts_ids[index+1]][kpts[parts_ids[index+1], 2]>self.conf_thr]
                    # print("next: ", next_part_kpts[:, 1])
                    next_max_part_y = np.amax(next_part_kpts[:, 1])

            # resize to 4 x 8 (width x height) same as the feature map size
            x1, y1, x2, y2 = 0, int((min_part_y-tl_y)/(br_y-tl_y)*self.vis_map_res[1]), self.vis_map_res[0], int((next_max_part_y-tl_y)/(br_y-tl_y)*self.vis_map_res[1])
            # print(x1, y1, x2, y2)
            visible_part_map[index, x1:x2, y1:y2] = 1
            visible_part_indicator[index] = 1
        # Front
        if self.params.use_ori:
            if ori > 90 and ori < 270:
                visible_indicator[:self.part_nums//2-1] = visible_part_indicator
                visible_indicator[self.part_nums//2-1] = 1
            # Back
            else:
                visible_indicator[self.part_nums//2:self.part_nums-1] = visible_part_indicator
                visible_indicator[self.part_nums-1] = 1
        else:
            visible_indicator[:self.part_nums-1] = visible_part_indicator
            visible_indicator[self.part_nums-1] = 1
        return maybe_cuda(visible_indicator.bool()), maybe_cuda(visible_part_map.bool())

    def get_ori(self, ori):
        """Segment the orientation
        Output:
            0 means the person not faces to the camera (Front)
            1 means the person faces to the camera (Back)
        """
        # Front
        if ori > 90 and ori < 270:
            return 0
        # Back
        else:
            return 1
    
    def crop_imgs(self, img, bbox):
        """Crop the images according to some bounding boxes. Typically for re-
        identification sub-module.
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            bboxes (Tensor): of shape (N, 4) or (N, 5).
        Returns:
            Tensor: Image tensor of shape (N, C, H, W).
        """
        x1, y1, x2, y2 = map(int, bbox)
        if x2 == x1:
            x2 = x1 + 1
        if y2 == y1:
            y2 = y1 + 1
        crop_img = img[:, :, y1:y2, x1:x2]
        if self.img_scale is not None:
            crop_img = F.interpolate(
                crop_img,
                size=self.img_scale,
                mode='bilinear',
                align_corners=False)
        return crop_img