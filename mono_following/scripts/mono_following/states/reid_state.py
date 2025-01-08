from states.state import State
from states.tracking_state import TrackingState
import rospy
from descriminator import Descriminator
class ReidState(State):
    def __init__(self):
        self.positive_count = {}

    def target(self):
        return -1

    def state_name(self):
        return "re-identification"
    
    def update(self, descriminator: Descriminator, tracks):
        descriminator.predict(tracks)  # return a dict
        for idx in tracks.keys():
            if tracks[idx].target_confidence ==None:
                continue
            if tracks[idx].target_confidence!=None and tracks[idx].target_confidence < rospy.get_param("~reid_pos_confidence_thresh", 0.2):
                continue
            
            if idx not in self.positive_count.keys():
                self.positive_count[idx] = 0
            self.positive_count[idx] += 1

            # consecutive verification
            if self.positive_count[idx] >= rospy.get_param("~reid_positive_count", 5):
                return TrackingState(idx)
        
        return self
