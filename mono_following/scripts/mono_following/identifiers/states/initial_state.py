from .state import State
from .initial_training_state import InitialTrainingState
from ..base_identifier import BaseIdentifier
# from ..params.hyper_params import HyperParams
class InitialState(State):
    def __init__(self, params):
        super().__init__(params)
        self.target_id = None
    
    def state_name(self):
        return "initialization"
    
    def update(self, identifier:BaseIdentifier, tracklets:dict):
        if identifier.target_id is None:
            return self
        self.target_id = identifier.target_id
        return InitialTrainingState(identifier.target_id, identifier.params)