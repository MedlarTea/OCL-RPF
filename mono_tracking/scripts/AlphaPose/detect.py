#! /usr/bin/env python3
import sys 
# sys.path.insert(0,'/home/hfy/cv_bridge_py3_ws/devel/lib/python3/dist-packages')
import os
import rospy
import ros_numpy
import cv2
from cv_bridge import CvBridge, CvBridgeError
import time
import torch
import numpy as np
from sensor_msgs.msg import Image
from mono_tracking.msg import Person, Persons, BodyPartElm
from DetectorLoader import TinyYOLOv3_onecls
# from mono_tracking.msg import BoxArray

from meters import AverageMeter

from YOLOX.detector import PersonDetector
from PoseEstimateLoader import SPPE_FastPose
from Track.Tracker import Detection
from fn import draw_single

_dir = os.path.split(os.path.realpath(__file__))[0]


class MonoDetector:
    def __init__(self):
        ###
        self.box_detector = PersonDetector(model='yolox-s', ckpt=os.path.join(_dir, 'YOLOX/weights/yolox_s.pth.tar'))
        ###
        ###
        # inp_dets = 384
        # self.box_detector = TinyYOLOv3_onecls(inp_dets, nms=0.2, conf_thres=0.4, device="cuda:0")
        ###
        self.joints_detector = SPPE_FastPose(backbone="resnet50", input_height=224, input_width=160, device="cuda:0")
        

        IMAGE_TOPIC = "/camera/color/image_raw"
        self.imageSub = rospy.Subscriber(IMAGE_TOPIC, Image, self.imageCallback, queue_size=1, buff_size=999999999)

        PERSONS_TOPIC = "/mono_detection/detections"
        self.personsPub = rospy.Publisher(PERSONS_TOPIC, Persons, queue_size=2)

        self.VISUALIZE_TOPIC = "/mono_detection/visualization"
        self.imageBoxesPub = rospy.Publisher(self.VISUALIZE_TOPIC, Image, queue_size=1)

        self.box_detect_time = AverageMeter()
        self.joints_detect_time = AverageMeter()
        self.once_time = AverageMeter()
        self.nums = 0
        self.print_freq = 100
        rospy.loginfo("MonoDetector is ready!")
        rospy.spin()
    
    def imageCallback(self, imgMsg):
        """
        In my RTX2060, time-0.05s(20Hz), detect-0.039/0.050, extract-0.009/0.050
        """
        # read image
        once_end = time.time()
        image = cv_bridge.imgmsg_to_cv2(imgMsg, "bgr8")
        # if imgMsg.encoding == "rgb8":
        #     image = ros_numpy.numpify(imgMsg)
        # elif imgMsg.encoding == "bgr8":
        #     image = ros_numpy.numpify(imgMsg)[:,:,[2,1,0]]  # change to rgb

        # define message to be sent
        persons = Persons()
        persons.header = imgMsg.header  # record original time, for distance estimation 
        # persons.image = imgMsg
        persons.image_h = image.shape[0]
        persons.image_w = image.shape[1]
    
        # box detect
        box_detect_end = time.time()
        ###
        xywhs, scores = self.box_detector.detect(image, conf=0.7)
        ###

        ###
        # detected = self.box_detector.detect(image, need_resize=True, expand_bb=10)
        ###
        self.box_detect_time.update((time.time()-box_detect_end)*1000)

        ###
        if len(xywhs) == 0:
            self.personsPub.publish(persons)
            return 
        ###

        # joints detect
        joints_detect_end = time.time()
        ###
        # if detected is not None:
        #     poses = self.joints_detector.predict(image, detected[:, 0:4], detected[:, 4])
        # else:
        #     self.personsPub.publish(persons)
        #     return 
        ###

        ###
        detected = self.xywh_to_xyxy_torch(xywhs, image.shape[1], image.shape[0])
        poses = self.joints_detector.predict(image, detected, torch.Tensor(scores))
        ###
        self.joints_detect_time.update((time.time()-joints_detect_end)*1000)

        # detection results
        detections = [Detection(ps['bbox'].numpy(),
                                    np.concatenate((ps['keypoints'].numpy(),
                                                    ps['kp_score'].numpy()), axis=1),
                                    ps['bbox_score']) for ps in poses]

        for i, detection in enumerate(detections):
            person = Person()
            bbox = detection.tlbr.astype(int)
            # [Nose, LEye, REye, LEar, REar, LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist, LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck]
            pts = np.concatenate((detection.keypoints, np.expand_dims((detection.keypoints[1, :] + detection.keypoints[2, :]) / 2, 0)), axis=0)
            for i, pt in enumerate(pts):
                joint = BodyPartElm()
                joint.part_id = i
                joint.x, joint.y, joint.confidence = pt[0], pt[1], pt[2]
                person.body_part.append(joint)
            person.box = bbox.tolist()
            person.score = detection.confidence
            persons.persons.append(person)
            # print(image.shape)
            # for drawing
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 3)  # draw bbox
            draw_single(image, detection.keypoints)

        self.personsPub.publish(persons)

        if self.imageBoxesPub.get_num_connections():
            # imgMsg = ros_numpy.msgify(Image, image, encoding="rgb8")
            imgMsg = cv_bridge.cv2_to_imgmsg(image, 'bgr8')
            self.imageBoxesPub.publish(imgMsg)
        

        self.once_time.update((time.time()-once_end)*1000)
        if((self.nums+1)%self.print_freq ==0):
            rospy.loginfo('BBoxTime {:.3f} ({:.3f})\t'
                          'KeysTime {:.3f} ({:.3f})\t'
                          'OnceTime {:.3f} ({:.3f})\t'.format(
                           self.box_detect_time.val, self.box_detect_time.avg,
                           self.joints_detect_time.val, self.joints_detect_time.avg,
                           self.once_time.val, self.once_time.avg
                        ))
        self.nums+=1
    
    def xywh_to_xyxy(self, bbox_xywh, img_width, img_height):
        x,y,w,h = bbox_xywh
        x1 = max(int(x-w/2),0)
        x2 = min(int(x+w/2),img_width-1)
        y1 = max(int(y-h/2),0)
        y2 = min(int(y+h/2),img_height-1)
        return x1,y1,x2,y2
    
    def visualize(self, image, boxes):
        visualized_image = image
        for box in boxes:
            x1,y1,x2,y2 = self.xywh_to_xyxy(box.box, image.shape[1], image.shape[0])
            visualized_image = cv2.rectangle(visualized_image,(x1,y1), (x2,y2), (255, 0, 0), 2)
        # imgMsg = ros_numpy.msgify(Image, visualized_image, encoding="rgb8")
        imgMsg = cv_bridge.cv2_to_imgmsg(visualized_image, 'bgr8')
        return imgMsg
        # self.imageBoxesPub.publish(imgMsg)
    
    def xywh_to_xyxy_torch(self, xywhs, img_width, img_height):
        """
        input:
            xywhs(N,4)
        """
        xyxys = []
        for xywh in xywhs:
            x1 = xywh[0]
            x2 = min(int(xywh[0]+xywh[2]),img_width-1)
            y1 = xywh[1]
            y2 = min(int(xywh[1]+xywh[3]),img_height-1)
            xyxys.append([x1, y1, x2, y2])

        return torch.Tensor(xyxys)


if __name__ == '__main__':
    rospy.init_node('mono_detector', anonymous=True)
    cv_bridge = CvBridge()
    monoDetector = MonoDetector()

