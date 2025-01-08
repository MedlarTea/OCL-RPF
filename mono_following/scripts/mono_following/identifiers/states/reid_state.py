from .state import State
from .tracking_state import TrackingState
from ..base_identifier import BaseIdentifier
# from ..params.hyper_params import HyperParams
from ..track_center.tracklet import Tracklet
class ReidState(State):
    def __init__(self, params, last_target_id):
        super().__init__(params)
        self.positive_count = {}
        self.last_target_id = last_target_id

    def target(self):
        return -1

    def state_name(self):
        return "re-identification"
    
    def update(self, identifier: BaseIdentifier, tracklets):
        identifier.predict(tracklets, self.state_name())
        for idx in tracklets.keys():
            # We want re-identify the target from stable bbox
            # if tracklets[idx].bbox_score < self.params.min_bbox_confidence:
            #     continue
            if tracklets[idx].target_confidence == None:
                continue
            # Find the target by the tracking module
            # if idx == self.last_target_id:
            #     return TrackingState(idx, identifier.params)
            
            if tracklets[idx].target_confidence < self.params.reid_pos_confidence_thresh:
                # consecutive check
                if idx in self.positive_count.keys():
                    self.positive_count[idx] = 0
                else:
                    continue
                continue
            
            if idx not in self.positive_count.keys():
                self.positive_count[idx] = 0
            self.positive_count[idx] += 1

            # consecutive verification
            if self.positive_count[idx] >= self.params.reid_positive_count:
                return TrackingState(idx, identifier.params)
        
        return self
