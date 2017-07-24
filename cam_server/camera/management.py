import socket
from logging import getLogger

from cam_server import config
from cam_server.camera.sender import process_camera_stream
from cam_server.instance_management.manager import InstanceManager
from cam_server.instance_management.wrapper import InstanceWrapper

_logger = getLogger(__name__)


class CameraInstanceManager(InstanceManager):
    def __init__(self, config_manager):
        super(CameraInstanceManager, self).__init__()

        self.config_manager = config_manager
        self.port_generator = iter(range(*config.CAMERA_STREAM_PORT_RANGE))

    def get_camera_list(self):
        return self.config_manager.get_camera_list()

    def get_camera_stream(self, camera_name):
        """
        Get the camera stream address.
        :param camera_name: Name of the camera to get the stream for.
        :return: Camera stream address.
        """

        # Check if the requested camera already exists.
        if not self.is_instance_present(camera_name):

            stream_port = next(self.port_generator)

            _logger.info("Creating camera instance '%s' on port %d.", camera_name, stream_port)

            self.add_instance(camera_name, CameraInstanceWrapper(
                process_function=process_camera_stream,
                camera=self.config_manager.load_camera(camera_name),
                stream_port=stream_port
            ))

        self.start_instance(camera_name)

        return self.get_instance(camera_name).stream_address


class CameraInstanceWrapper(InstanceWrapper):
    def __init__(self, process_function, camera, stream_port):

        super(CameraInstanceWrapper, self).__init__(camera.get_name(), process_function,
                                                    camera, stream_port)

        self.camera = camera
        self.stream_address = "tcp://%s:%d" % (socket.gethostname(), stream_port)

    def get_info(self):
        return {"stream_address": self.stream_address,
                "is_stream_active": self.is_running(),
                "camera_geometry": self.camera.get_geometry(),
                "camera_name": self.camera.get_name()}

    def get_name(self):
        return self.camera.get_name()
