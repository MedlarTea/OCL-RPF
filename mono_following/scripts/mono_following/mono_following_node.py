#! /usr/bin/env python3
import numpy as np
import sys
import os
print("File position:", os.getcwd())
print("Python exe:", sys.executable)
# print(sys.path)
import ros_numpy
import cv2
import rospy
import os.path as osp

# My class
from tracklet import Tracklet
from states.initial_state import InitialState
from descriminator import Descriminator
# standard ROS message
from sensor_msgs.msg import Image
# import tf

# my ROS message
from spencer_tracking_msgs.msg import TrackedPerson, TrackedPersons, TargetPerson, Box
from mono_tracking.msg import TrackArray
from mono_tracking.msg import Track
from mono_following.msg import Target
import message_filters
# from mono_following.msg import Box
from mmtrack.apis import init_model, inference_rpf_wo_gt
from mmtrack.utils.meters import AverageMeter
import mmcv

import time


class MonoFollowing:
    def __init__(self):
        ### Some parameters for debug and testing ###
        self.evaluate = rospy.get_param("~evaluate")
        if self.evaluate:
            self.STORE_DIR = rospy.get_param("~track_store_dir")
            self.result_fname = osp.join(self.STORE_DIR, "Visible-Parts-MPF.txt")
            print("Store result to: {}".format(self.result_fname))
            self.result_fp = open(self.result_fname, "w")
            self.num = 0

        self.oclreid_dir = rospy.get_param("~oclreid_dir")

        imageTopic = "/camera/color/image_raw"
        self.imageSubscriber = message_filters.Subscriber(imageTopic, Image)
        trackedPersonsTopic = "/mono_tracking/tracks"
        tracks_sub = message_filters.Subscriber(trackedPersonsTopic, TrackedPersons)
        tss = message_filters.ApproximateTimeSynchronizer([tracks_sub, self.imageSubscriber], queue_size=20, slop=0.8)
        tss.registerCallback(self.callback)

        # init the discriminator
        self.init_discriminator()
        # record time
        self.reid_time = AverageMeter()

        # publish image for visualization
        self.image_pub = rospy.Publisher("/mono_following/vis_image", Image, queue_size=1)
        # publish image patches for visualization
        self.patches_pub = rospy.Publisher("/mono_following/patches", Image, queue_size=1)
        # publish target information
        self.target_pub = rospy.Publisher("/mono_following/target", TargetPerson, queue_size=1)

        ### Node is already set up ###
        rospy.loginfo("Mono Following Node is Ready!")
        rospy.spin()
    
    def init_discriminator(self):
        self.target_id = None
        self.frame_id = 0
        self.device = "cuda:0"
        hyper_params = mmcv.Config.fromfile(osp.join(self.oclreid_dir, "tpt_configs/run_robot_hyper_params.py"))
        hyper_params.mmtracking_dir = self.oclreid_dir


        rpf_config = osp.join(self.oclreid_dir, hyper_params.rpf_config)
        rpf_config = mmcv.Config.fromfile(rpf_config)
        rpf_config.model.reid.init_cfg.checkpoint = osp.join(self.oclreid_dir, "checkpoints/reid/resnet18.pth")

        identifier_config = osp.join(self.oclreid_dir, hyper_params.identifier_config)
        identifier_config = mmcv.Config.fromfile(identifier_config)


        self.discriminator = init_model(rpf_config, None, device=self.device, hyper_config=hyper_params, identifier_config=identifier_config, seed=123)

    def cal_dist(self, person_pose):
        return np.linalg.norm([person_pose.pose.position.x, person_pose.pose.position.y, person_pose.pose.position.z])

    def init_target(self, track_ids, track_dists):
        if len(track_ids) >0:
            return track_ids[0]
        else:
            return None

    def callback(self, tracks_msg, image_msg):
        print("---------- REID CALLBACK ----------")
        # get messages
        if image_msg.encoding == "rgb8":
            image = ros_numpy.numpify(image_msg)
        elif image_msg.encoding == "bgr8":
            image = ros_numpy.numpify(image_msg)[:,:,[2,1,0]]  # change to rgb

        track_ids = []
        track_bboxes = []
        track_joints = []
        track_dists = []
        track_msgs = {}
        for track in tracks_msg.tracks:
            if track.pose.pose.position.x == 0 or track.is_matched == False or track.bounding_box.w == 0:
                continue
            # print(track)
            ### Construct information for mmtrack ###
            track_msgs[track.track_id] = track
            track_dists.append(self.cal_dist(track.pose))
            x,y,w,h,confidence = (track.bounding_box.x, track.bounding_box.y, track.bounding_box.w, track.bounding_box.h, track.bounding_box.confidence)
            track_bboxes.append([x, y, x+w, y+h])
            track_ids.append(track.track_id)
            # do not use neck joint
            t_joints = []
            for i in range(len(track.body_parts)-1):
                body_part = track.body_parts[i]
                t_joints.append([body_part.x, body_part.y, body_part.confidence])  # (13,3)
            track_joints.append(t_joints)  # (N,13,3)
        # print(track_bboxes)
        # print(track_joints)

        ### select target ###
        if self.target_id is None:
            self.target_id = self.init_target(track_ids, track_dists)
        
        # print("track_ids:", track_ids)
        ### use mmtrack's identifier to find the target ###
        t=time.time()
        ident_result = inference_rpf_wo_gt(self.discriminator, image, track_ids, track_bboxes, track_joints, self.target_id, self.frame_id)
        self.reid_time.update((time.time()-t)*1000)
        print("[REID Time]  Current {:.3f}\tAverage {:.3f}".format(self.reid_time.val, self.reid_time.avg))
        if len(ident_result) != 0:
            self.state = ident_result["state"]
            self.target_id = ident_result["target_id"]
            self.frame_id += 1
        ### use mmtrack's identifier to find the target ###

        # buffer_imgs = ident_result.get("buffer_imgs", None)
        # if buffer_imgs is not None:
        #     for par_id in buffer_imgs.keys():
        #         for y_int in buffer_imgs[par_id].keys():
        #             bu_imgs = buffer_imgs[par_id][y_int]  # torch.tensor
        #             print(bu_imgs.shape)
        #             print(par_id, y_int)
        #             _dir = osp.join(self.target_image_path, "buffer_imgs", str(par_id)+"_"+str(y_int))
        #             if not os.path.exists(_dir):
        #                 os.makedirs(_dir)
        #             for j in range(bu_imgs.shape[0]):
        #                 bu_img = bu_imgs[j].permute(1, 2, 0).cpu().numpy()
        #                 if bu_img.size == 0:
        #                     continue
        #                 # print(image.shape)
        #                 bu_img_bgr = cv2.cvtColor(bu_img, cv2.COLOR_RGB2BGR)
        #                 cv2.imwrite(osp.join(_dir, "{:04d}.jpg".format(j)), bu_img_bgr)
        
        # publish target information
        target = TargetPerson()
        target.header = tracks_msg.header
        
        for idx in track_msgs.keys():
            if idx == self.target_id:
                target_confidence = ident_result["tracks_target_conf_bbox"][idx][1]
                # print(target_confidence)
                target.track_id = self.target_id
                target.bounding_box = track_msgs[idx].bounding_box if target_confidence != None else Box()
                target.target_confidence = target_confidence if target_confidence != None else 0.0
                target.pose = track_msgs[idx].pose
                target.twist = track_msgs[idx].twist
                target.is_matched = True

        self.target_pub.publish(target)

        # save result
        # if self.evaluate:
        #     self.save_result(target.bounding_box)
        
        # publish image containing the identification result
        if self.image_pub.get_num_connections():
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            tracks_result = ident_result["tracks_target_conf_bbox"] if len(ident_result) != 0 else {}
            image_bgr, target_image = self.visualize(image_bgr, tracks_result)
            # if target_image is not None:
                # cv2.imwrite(osp.join(self.target_image_path, "{:04d}.jpg".format(self.frame_id)), target_image)
            # cv2.imwrite(osp.join(self.image_path, "{:04d}.jpg".format(self.frame_id)), image_bgr)
            image_msg = ros_numpy.msgify(Image, image_bgr, encoding = "bgr8")
            self.image_pub.publish(image_msg)

        # publish target patch to see whether is OK
        # if self.patches_pub.get_num_connections():
        #     image = self.visualize_patches(track_msgs)
        #     if image is not None:
        #         # cv2.imwrite("./test.jpg", image)
        #         image_msg = ros_numpy.msgify(Image, image, encoding = "rgb8")
        #         self.patches_pub.publish(image_msg)

        # Save tracks information, for debug and analysis

    def visualize(self, original_image, tracks):
        # print("VISUALIZE")
        image = original_image.copy()
        target_img = None
        for idx in tracks.keys():
            if tracks[idx][0] == None:
                continue
            image = cv2.putText(original_image, self.state, (20, 60), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
            if idx == self.target_id:
                target_img = original_image.copy()
                target_img = target_img[tracks[idx][2][1]:tracks[idx][2][3], tracks[idx][2][0]:tracks[idx][2][2], :]
                image = cv2.rectangle(image, (tracks[idx][2][0],tracks[idx][2][1]), (tracks[idx][2][2],tracks[idx][2][3]), (0,255,0), 3)
                image = cv2.putText(image, "id:{:d}".format(idx), ((int((tracks[idx][2][0]+tracks[idx][2][2])/2-5), int((tracks[idx][2][1]+tracks[idx][2][3])/2-5))), cv2.FONT_HERSHEY_DUPLEX, 1, (0,255,0), 1)
                # image = cv2.putText(image, "score:{:.3f}".format(tracks[idx].target_confidence), ((int((tracks[idx].region[0]+tracks[idx].region[2])/2-35), int((tracks[idx].region[1]+tracks[idx].region[3])/2-35))), cv2.FONT_HERSHEY_DUPLEX, 1, (0,255,0), 1)
            else:
                image = cv2.rectangle(image, (tracks[idx][2][0],tracks[idx][2][1]), (tracks[idx][2][2],tracks[idx][2][3]), (0,0,255), 3)
                image = cv2.putText(image, "id:{:d}".format(idx), ((int((tracks[idx][2][0]+tracks[idx][2][2])/2-5), int((tracks[idx][2][1]+tracks[idx][2][3])/2-5))), cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,255), 1)
                # image = cv2.putText(image, "score:{:.3f}".format(tracks[idx].target_confidence), ((int((tracks[idx].region[0]+tracks[idx].region[2])/2-35), int((tracks[idx].region[1]+tracks[idx].region[3])/2-35))), cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,255), 1)

        return image, target_img
    
    def saveTrack(self, track_id, track, store_path):
        print("SAVE TRACKS")

    # def visualize_patches(self, tracks):
    #     target_id = self.target_id
    #     if target_id in tracks.keys() and tracks[target_id].image_patch is not None:
    #         return tracks[target_id].image_patch
    #     else:
    #         return None
    
    def save_result(self, bbox:Box):
        """Save target bbox result for evaluation, this is for icvs dataset.
        Input:
            bbox messgae: [x1, y1, w, h]
        Write:
            bbox as [x1, y1, w, h]
        """
        self.result_fp.write("{} {} {} {} {}\n".format(self.num, int(bbox.x), int(bbox.y), int(bbox.w), int(bbox.h)))
        self.num += 1

if __name__ == "__main__":
    rospy.init_node('mono_following', anonymous=True)
    mono_following = MonoFollowing()

