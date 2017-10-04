import socket
from logging import getLogger

from cam_server import config
from cam_server.camera.sender import get_sender_function
from cam_server.instance_management.management import InstanceManager, InstanceWrapper
from cam_server.utils import get_port_generator

_logger = getLogger(__name__)


class CameraInstanceManager(InstanceManager):
    def __init__(self, config_manager, hostname=None):
        super(CameraInstanceManager, self).__init__()

        self.config_manager = config_manager
        self.port_generator = get_port_generator(config.CAMERA_STREAM_PORT_RANGE)
        self.hostname = hostname

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

            camera = self.config_manager.load_camera(camera_name)
            camera.verify_camera_online()

            _logger.info("Creating camera instance '%s' on port %d.", camera_name, stream_port)

            camera_config = self.config_manager.get_camera_config(camera_name)

            self.add_instance(camera_name, CameraInstance(
                process_function=get_sender_function(camera_config.get_source_type()),
                camera=camera,
                stream_port=stream_port,
                hostname=self.hostname
            ))

        self.start_instance(camera_name)

        return self.get_instance(camera_name).get_stream_address()

    def set_camera_instance_config(self, camera_name, new_config):
        if not self.is_instance_present(camera_name):
            return

        camera_instance = self.get_instance(camera_name)
        camera_instance.set_parameter(new_config)


class CameraInstance(InstanceWrapper):
    def __init__(self, process_function, camera, stream_port, hostname=None):

        super(CameraInstance, self).__init__(camera.get_name(), process_function,
                                             camera, stream_port)

        self.camera = camera

        if not hostname:
            hostname = socket.gethostname()

        self.stream_address = "tcp://%s:%d" % (hostname, stream_port)

    def get_info(self):
        return {"stream_address": self.stream_address,
                "is_stream_active": self.is_running(),
                "camera_geometry": self.camera.get_geometry(),
                "camera_name": self.camera.get_name(),
                "last_start_time": self.last_start_time,
                "statistics": self.get_statistics(),
                "config": self.get_configuration()}

    def get_configuration(self):
        return self.camera.camera_config.get_configuration()

    def get_name(self):
        return self.camera.get_name()

    def get_stream_address(self):
        return self.stream_address

    def set_parameter(self, configuration):
        self.camera.camera_config.set_configuration(configuration)

        # The set configuration sets the default parameters.
        super().set_parameter(self.camera.camera_config.get_configuration())
