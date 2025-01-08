from .state import State
from ..base_identifier import BaseIdentifier
# from ..params.hyper_params import HyperParams
from .tracking_state import TrackingState

class InitialTrainingState(State):
    def __init__(self, target_id, params):
        super().__init__(params)
        self.target_id = target_id
        self.num_pos_samples = 0
    
    def target(self):
        return self.target_id

    def state_name(self):
        return "initial training"
    
    def update(self, identifier: BaseIdentifier, tracklets: dict):
        # print("Initial training")
        from .initial_state import InitialState
        if self.target_id not in tracklets.keys():
            return InitialState(identifier.params)

        ### update memory ###
        is_memory_updated = identifier.update_memory(tracklets, self.target_id)
        
        ### update classifier ###
        identifier.newest_st_loss, identifier.newest_lt_loss = identifier.update_classifier()

        if is_memory_updated:
            self.num_pos_samples += 1
        if self.num_pos_samples >= self.params.initial_training_num_samples:
            return TrackingState(self.target_id, identifier.params)
        return self
    

