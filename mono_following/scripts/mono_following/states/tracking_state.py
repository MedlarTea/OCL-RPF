from states.state import State
import sys
sys.path.append("..")
from descriminator import Descriminator
import rospy

class TrackingState(State):
    def __init__(self, target_id):
        self.target_id = target_id
    
    def target(self):
        return self.target_id
        
    def state_name(self):
        return "tracking"
    
    def update(self, descriminator: Descriminator, tracks: dict):
        from states.reid_state import ReidState
        if self.target_id not in tracks.keys():
            return ReidState()
        
        descriminator.predict(tracks)
        pred = tracks[self.target_id].target_confidence

        if pred!=None and pred < rospy.get_param('~id_switch_detection_thresh', default=-1.0):
            rospy.loginfo("ID switch detected!!")
            return ReidState()
        # This target is not convisible
        if pred==None or pred < rospy.get_param("~min_target_confidence", -1):
            rospy.loginfo("do not update")
            return self
        
        isSuccessed = descriminator.updateFeatures(tracks, self.target_id)

        return self
    

