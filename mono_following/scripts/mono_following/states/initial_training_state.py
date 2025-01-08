from states.state import State
from descriminator import Descriminator
from states.tracking_state import TrackingState
import rospy

class InitialTrainingState(State):
    def __init__(self, track_id):
        self.target_id = track_id
        self.num_pos_samples = 0
    
    def target(self):
        return self.target_id

    def state_name(self):
        return "initial training"
    
    def update(self, descriminator: Descriminator, tracks: dict):
        print("Initial training")
        from states.initial_state import InitialState
        if self.target_id not in tracks.keys():
            return InitialState()
        isSuccessed = descriminator.updateFeatures(tracks, self.target_id)
        if isSuccessed:
            self.num_pos_samples += 1
        if self.num_pos_samples >= rospy.get_param("~initial_training_num_samples"):
            return TrackingState(self.target_id)
        return self
    

