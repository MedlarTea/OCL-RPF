from states.state import State
from states.initial_training_state import InitialTrainingState
class InitialState(State):
    def __init__(self):
        pass
    
    def state_name(self):
        return "initialization"
    
    def update(self, descriminator, tracks):
        target_id = self.select_target(tracks)
        if target_id < 0:
            return self
        return InitialTrainingState(target_id)

    def select_target(self, tracks: dict) -> State:
        target_id = -1
        distance = 0.0

        for id in tracks.keys():
            pos = tracks[id].pos_in_baselink  # [x,y]
            current_dis = tracks[id].distance
            if (len(tracks.keys())!=1 and abs(pos[1])>0.2):
                continue
            if(target_id==-1 or distance > current_dis):
                target_id = id
                distance = current_dis
        return target_id