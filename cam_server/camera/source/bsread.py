from logging import getLogger

from cam_server.camera.source.epics import CameraEpics

_logger = getLogger(__name__)


class CameraBsread(CameraEpics):
    def __init__(self, camera_config):
        super(CameraBsread, self).__init__(camera_config)

    def verify_camera_online(self):
        pass

    def connect(self):
        pass

    def disconnect(self):
        pass

    def get_image(self):
        pass

    def get_geometry(self):
        pass
