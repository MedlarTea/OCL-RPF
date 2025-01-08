from abc import ABC, abstractmethod
from ..params.hyper_params import HyperParams

class State(ABC):
    def __init__(self, params):
        # super(HyperParams, self).__init__(params) 
        self.params = params
    
    def target(self):
        return -1
    
    @abstractmethod
    def state_name(self):
        pass

    @abstractmethod
    def update(self, identifier, tracklets):
        pass
