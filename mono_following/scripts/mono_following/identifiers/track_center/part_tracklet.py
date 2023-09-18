import cv2
import torch
import torch.nn.functional as F
import numpy as np

from scipy.special import softmax
from mmtrack.models.identifier.utils.utils import maybe_cuda
"""
Tracklet
1) use multi threads to accelerate single->batch and batch->single procedure (for deep-learning-based feature extractor)
"""
class PartTracklet:
    def __init__(self, img, img_metas, params, observation, rescale=False):
        """
        image (numpy.array): original image with (H,W,3)
        bbox (list(int)): [tl_x, tl_y, br_x, br_y]
        """
        self.params = params
        self.vis_map_size = params.vis_map_size
        self.vis_map_nums = self.vis_map_size[0]
        self.vis_map_res = self.vis_map_size[1:]
        self.part_nums = self.vis_map_nums+1 if not params.use_ori else 2*(self.vis_map_nums+1)
        ### prior parameters ### 
        self.img_scale = (256, 128)
        self.conf_thr = 0.3
        self.temp = 0.1
        # self.down_width = 64
        # self.down_height = 128

        ### information ###
        # self.track_id = track[0]
        self.bbox = observation[:4]  # Tensor
        self.image_patch = self.crop_imgs(img=img, 
                                          img_metas=img_metas,
                                          bbox=self.bbox, 
                                          rescale=rescale) # Tensor
        self.bbox_feature = self.get_bbox_feature(img_metas=img_metas, bbox=self.bbox) 


        # print("\nobservation: ", observation)
        if len(observation) > 4:
            self.bbox_score = observation[4]
            if self.params.use_joint:
                self.kpts = observation[5]
                # [Nose, LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist, LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck]
                self.kpts = np.concatenate((self.kpts, np.expand_dims((np.array(self.kpts)[1, :] + np.array(self.kpts)[2, :]) / 2, 0)), axis=0)
                self.joints_feature = self.get_joints_feature(img_metas, self.kpts)
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

    
    def get_bbox_feature(self, img_metas, bbox):
        """get box information
        Input:
            box: List of (4) with x1, y1, x2, y2
        Output:
            box's scaled height, box's scaled width
        """
        h, w, _ = img_metas[0]['img_shape']
        x1, y1, x2, y2 = map(int, bbox)
        bbox_width = abs(x1-x2)
        bbox_height = abs(y1-y2)
        return bbox_height/h, bbox_width/w
    
    def get_joints_feature(self, img_metas, kpts: np.array):
        """get joints information
        Input:
            kpts: [Nose, LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist, LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck]
        Output:
            Scaled coordinates of joints
        """
        h, w, _ = img_metas[0]['img_shape']
        scaled_kpts = np.zeros((kpts.shape[0], 2))
        scaled_kpts[:, 0] = kpts[:, 0] / w
        scaled_kpts[:, 1] = kpts[:, 1] / h
        mask = kpts[:,2] > self.conf_thr
        # print(scaled_kpts)
        # print(mask)
        scaled_kpts = scaled_kpts * np.expand_dims(mask, axis=1)
        # print(scaled_kpts)
        scaled_kpts = scaled_kpts.flatten()
        scaled_kpts = softmax(scaled_kpts/self.temp)
        return maybe_cuda(torch.Tensor(scaled_kpts))

    def down_scale(self, image, bbox):
        x1,y1,x2,y2 = bbox
        # print(x1,y1,x2,y2)
        image = image[y1:y2, x1:x2, :]
        image = cv2.resize(image, (self.down_width, self.down_height), interpolation=cv2.INTER_LINEAR)
        return image
    
    def crop_imgs(self, img, img_metas, bbox, rescale=False):
        """Crop the images according to some bounding boxes. Typically for re-
        identification sub-module.

        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
            bboxes (Tensor): of shape (N, 4) or (N, 5).
            rescale (bool, optional): If True, the bounding boxes should be
                rescaled to fit the scale of the image. Defaults to False.

        Returns:
            Tensor: Image tensor of shape (N, C, H, W).
        """
        h, w, _ = img_metas[0]['img_shape']
        img = img[:, :, :h, :w]
        
        ### TODO: remove in the future###
        if not isinstance(bbox, torch.Tensor):
            bbox = torch.Tensor(bbox).to(img.device)

        if rescale:
            bbox[:4] *= torch.tensor(img_metas[0]['scale_factor']).to(
                bbox.device)
        bbox[0::2] = torch.clamp(bbox[0::2], min=0, max=w)
        bbox[1::2] = torch.clamp(bbox[1::2], min=0, max=h)

        # crop_imgs = []
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

class PartTracklet_for_predict:
    def __init__(self, img_metas, image_patch, deep_feature, kpts, visibility_indicator, binary_ori):
        self.img_metas = img_metas
        self.img_patch = image_patch
        self.deep_feature = deep_feature
        self.kpts = kpts.cpu().numpy()
        self.visibility_indicator = visibility_indicator
        self.binary_ori = binary_ori
        self.joints_feature = self.get_joints_feature(self.img_metas, self.kpts)
        self.target_confidence = None
        self.part_target_confidence = [None for _ in range(len(visibility_indicator))]
        
    
    def get_joints_feature(self, img_metas, kpts: np.array, conf_thr=0.3, temp=0.1):
        """get joints information
        Input:
            kpts: [Nose, LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist, LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck]
        Output:
            Scaled coordinates of joints
        """
        h, w, _ = img_metas[0]['img_shape']
        scaled_kpts = np.zeros((kpts.shape[0], 2))
        scaled_kpts[:, 0] = kpts[:, 0] / w
        scaled_kpts[:, 1] = kpts[:, 1] / h
        mask = kpts[:,2] > conf_thr
        # print(scaled_kpts)
        # print(mask)
        scaled_kpts = scaled_kpts * np.expand_dims(mask, axis=1)
        # print(scaled_kpts)
        scaled_kpts = scaled_kpts.flatten()
        scaled_kpts = softmax(scaled_kpts/temp)
        return maybe_cuda(torch.Tensor(scaled_kpts))