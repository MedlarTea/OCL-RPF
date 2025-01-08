import rospy
from mono_tracking.msg import Person, Persons, BodyPartElm
from sensor_msgs.msg import CameraInfo
from spencer_tracking_msgs.msg import TrackedPerson, TrackedPersons

class MonoTrackingNode():
    def __init__(self,):
        OBSERVATION_TOPIC = "/people/detections"
        self.observationSub = rospy.Subscriber(OBSERVATION_TOPIC, Persons, self.observation_callback, queue_size=2)

        # CAMERA_INFO_TOPIC = "/camera/color/camera_info"
        # self.cameraInfoSub = rospy.Subscriber(CAMERA_INFO_TOPIC, CameraInfo, self.cameraInfo_callback, queue_size=2)

        TRACKS_TOPIC = "/people/tracks"
        self.tracks_pub = rospy.Publisher(TRACKS_TOPIC, TrackedPersons, queue_size=10)

        VISUALIZATION_TOPIC = "/people/visualization"
        
        rospy.loginfo("MonoTracker is ready!")
        rospy.spin()

    # def cameraInfo_callback(self, cameraInfo_msg):
    #     pass

    def observation_callback(self, persons_msg):
        observations = []
        for person in persons_msg.persons:
            pass

    def create_people_msgs(self, now_time, image):
        pass

    def visualize(self):
        pass

