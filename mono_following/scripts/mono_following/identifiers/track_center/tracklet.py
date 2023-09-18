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
class Tracklet:
    def __init__(self, img, img_metas, observation, rescale=False):
        """
        image (numpy.array): original image with (H,W,3)
        bbox (list(int)): [tl_x, tl_y, br_x, br_y]
        """
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

        if len(observation) > 4:
            self.bbox_score = observation[4]
            self.kpts = observation[-1]
            # [Nose, LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist, LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck]
            self.kpts = np.concatenate((self.kpts, np.expand_dims((np.array(self.kpts)[1, :] + np.array(self.kpts)[2, :]) / 2, 0)), axis=0)
            self.joints_feature = self.get_joints_feature(img_metas, self.kpts)
            # print("\nkpts: {}".format(self.kpts))

        ### feature ###
        self.descriptor = None
        self.deep_feature = None
        self.target_confidence = None
    
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

class Tracklet_for_predict:
    def __init__(self, img_metas, image_patch, deep_feature, kpts):
        self.img_metas = img_metas
        self.img_patch = image_patch
        self.deep_feature = deep_feature
        self.kpts = kpts.cpu().numpy()
        self.joints_feature = self.get_joints_feature(self.img_metas, self.kpts)
        self.target_confidence = None
    
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
    