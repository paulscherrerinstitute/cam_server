import numpy
import time
import h5py
import logging
from cam_server.camera.source.camera import *

_logger = logging.getLogger(__name__)


class Hdf5(Camera):
    def __init__(self, camera_config):
        Camera.__init__(self, camera_config)
        self.camera_config = camera_config
        source = camera_config.get_source()
        tokens = source.rsplit(':', 1)
        self.file_name = tokens[0]
        self.path = tokens[1]
        self.shape = None
        self.check_data = True
        self.interval = camera_config.get_configuration().get("interval", 1.0)

    def get_raw_geometry(self):
        return self.width_raw, self.height_raw

    def verify_camera_online(self):
        pass

    def connect(self):
        self.h5 = h5py.File(self.file_name, 'r')
        self.dataset =self.h5[self.path]
        _logger.info("Connecting to "  +str(self.file_name) + ":" + str(self.path) + " - " + str(self.dataset))
        self.shape = self.dataset.shape # (4,100,200)
        # Width and height of the raw image
        self.width_raw = self.shape[2]
        self.height_raw = self.shape[1]
        self.num_images = self.shape[0]
        self.cur_index = 0

    def disconnect(self):
        pass

    def read(self):
        ret =self.dataset[self.cur_index]
        self.cur_index = self.cur_index+1
        if self.cur_index >=self.num_images:
            self.cur_index = 0
        return ret