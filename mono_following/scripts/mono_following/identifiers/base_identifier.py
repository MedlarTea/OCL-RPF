import os
import numpy as np
from abc import ABCMeta, abstractmethod


class BaseIdentifier(metaclass=ABCMeta):
    """Base class for target person identifier."""

    def __init__(self):
        super(BaseIdentifier, self).__init__()
        self.target_id = None
        self.visdom = None
        self.params = None
        self.frame_id = 0
        self.newest_loss = -1
    
    @abstractmethod
    def identify(self, *args, **kwargs):
        """Identify the target person."""
        pass

    @abstractmethod
    def extract_features(self, frame_id, image, tracks: dict):
        """extract image patches features based on the tracks information
        frame_id (int): frame id
        image (array): with shape (3, width, height)
        tracks (dict): {id(int):bbox[tl_x, tl_y, br_x, br_y]}
        """
        pass
    
    @abstractmethod
    def update_memory(self, tracklets: dict, target_id: int):
        pass
    
    @abstractmethod
    def update_classifier(self):
        """
        return loss
        """
        pass
    
    @abstractmethod
    def predict(self, tracklets: dict, state):
        pass