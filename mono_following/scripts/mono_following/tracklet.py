from geometry_msgs.msg import PointStamped
import cv2
import numpy as np
from spencer_tracking_msgs.msg import TrackedPerson

class Tracklet:
    def __init__(self, header, track_msg:TrackedPerson, image):
        ### prior parameters ### 
        self.down_width = 64
        self.down_height = 128

        ### information ###
        self.region = None
        self.image_patch = None
        self.descriptor = None
        self.target_confidence = None

        ### transform the point ###
        self.person_poseWithCovariance = track_msg.pose
        self.twist = track_msg.twist
        self.joints = track_msg.body_parts
        self.pos_in_baselink = [self.person_poseWithCovariance.pose.position.x, self.person_poseWithCovariance.pose.position.y, self.person_poseWithCovariance.pose.position.z]
        self.distance = np.linalg.norm(self.pos_in_baselink)

        # print(track_msg.box.box)
        if track_msg.bounding_box.w!=0 and track_msg.bounding_box.h!=0:
            # u_tl, v_tl, u_br, v_br
            self.bounding_box = track_msg.bounding_box
            x,y,w,h = (track_msg.bounding_box.x, track_msg.bounding_box.y, track_msg.bounding_box.w, track_msg.bounding_box.h)
            self.region = [x, y, x+w, y+h]
            self.image_patch = self.preprocessImage(image, self.region)
            # print(self.image_patch.shape)

    def preprocessImage(self, image, region):
        x1,y1,x2,y2 = region
        # print(x1,y1,x2,y2)
        image = image[y1:y2, x1:x2, :]
        image = cv2.resize(image, (self.down_width, self.down_height), interpolation=cv2.INTER_LINEAR)
        return image
    


