from .state import State
import sys
sys.path.append("..")
from ..base_identifier import BaseIdentifier
# import rospy

class TrackingState(State):
    def __init__(self, target_id, params):
        super().__init__(params)
        self.target_id = target_id
    
    def target(self):
        return self.target_id
        
    def state_name(self):
        return "tracking"
    
    def update(self, identifier: BaseIdentifier, tracklets: dict):
        from .reid_state import ReidState
        if self.target_id not in tracklets.keys():
            print("Cannot find the target!!")
            return ReidState(identifier.params, self.target_id)
        
        identifier.predict(tracklets, self.state_name())
        pred = tracklets[self.target_id].target_confidence
        # print("target conf: {:.3f}".format(pred))

        if pred!=None and pred < self.params.id_switch_detection_thresh:
            print("ID switch detected!!")
            return ReidState(identifier.params, -1)
        # This target is not convisible
        if pred==None or pred < self.params.min_target_confidence:
            print("do not update")
            return self
        
        ### update memory ###
        identifier.incremental_st_loss = identifier.update_memory(tracklets, self.target_id)
        
        ### update classifier ###
        identifier.newest_st_loss, identifier.newest_lt_loss = identifier.update_classifier()

        return self
    

