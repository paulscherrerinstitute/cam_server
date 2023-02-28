from cam_server.camera.source.camera import *

_logger = getLogger(__name__)


class custom_source(Camera):
    def __init__(self, camera_config):
        Camera.__init__(self, camera_config)
        self.camera_config = camera_config
        # Width and height of the raw image
        self.width_raw = 80
        self.height_raw = 40

    def get_raw_geometry(self):
        return self.width_raw, self.height_raw

    def verify_camera_online(self):
        return

    def connect(self):
        pass

    def disconnect(self):
        pass

    def read(self):
        width, height = self.get_raw_geometry()
        return numpy.random.randint(1, 101,  width * height, "uint16").reshape((height, width))

    def process(self, stop_event, statistics, parameter_queue, logs_queue, port):
        return Camera.process(self, stop_event, statistics, parameter_queue, logs_queue, port)
