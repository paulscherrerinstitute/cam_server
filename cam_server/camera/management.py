import socket
from logging import getLogger

from cam_server import config
from cam_server.camera.source.camera import get_ipc_address
from cam_server.camera.sender import get_sender_function
import cam_server.camera.source.utils as utils
from cam_server.camera.source.utils import get_source_class
from cam_server.instance_management.management import InstanceManager, InstanceWrapper
from cam_server.instance_management.configuration import TempBackgroundImageManager

_logger = getLogger(__name__)


class CameraInstanceManager(InstanceManager):
    #Mode: 0 (default): When an instance stops, it is not deleted: the process, camera object and port.
    #                   are reused if the instance is restarted.
    #                   Limitation: CameraWorker must be restarted when camera type changes (EPICS->BSREAD)
    #                               as camera object is not re-instantiated.
    #Mode: 1: When an instance stops, keeps its process, camera object and port but when restarted checks if  the camera
    #         type changed and, in this case, stops and delete the instance so a new camera object is created.
    #Mode: 2: Same as 1 but tries to reuse the same port as the last used by the camera.
    #Mode: 3: Automatically deletes stopped instances (as pipelines do).
    #         Every time an instance is restarted it gets a new process, camera object and port.
    #Mode: 4: Same as 4 but tries to reuse the same port as the last used by the camera.

    # default, allow_reinstantiate, auto_delete, auto_delete_prefer_same_port
    def __init__(self, config_manager, user_scripts_manager, hostname=None, port_range=None, mode=0):
        super(CameraInstanceManager, self).__init__(
            port_range=config.CAMERA_STREAM_PORT_RANGE if (port_range is None) else port_range,
            auto_delete_stopped=(mode in (3, 4)))
        self.prefer_same_port = mode not in (1, 3)
        self.allow_reinstantiate = (mode == 1)
        self.config_manager = config_manager
        self.user_scripts_manager = user_scripts_manager
        self.hostname = hostname
        self.background_manager = TempBackgroundImageManager(clear=True)
        utils._user_scripts_manager = user_scripts_manager

    def get_camera_list(self):
        return self.config_manager.get_camera_list()

    def get_instance_stream(self, camera_name):
        """
        Get the camera stream address.
        :param camera_name: Name of the camera to get the stream for.
        :return: Camera stream address.
        """
        # Check if the requested camera already exists.
        if self.allow_reinstantiate and self.is_instance_present(camera_name):
            try:
                if type(self.get_instance(camera_name).camera) != \
                        get_source_class(self.get_instance(camera_name).get_configuration()):
                    self.stop_instance(camera_name)
                    self.delete_stopped_instance(camera_name)
            except:
                pass

        if not self.is_instance_present(camera_name):
            camera = self.config_manager.load_camera(camera_name)
            camera.verify_camera_online()
            camera_config = self.config_manager.get_camera_config(camera_name)

            if camera_config.get_configuration().get("port"):
                stream_port = int(camera_config.get_configuration().get("port"))
                _logger.info("Creating camera stream on fixed port '%s' for camera '%s" %
                             (stream_port, camera_name))
            else:
                stream_port = self.get_next_available_port(camera_name, self.prefer_same_port)
                _logger.info("Creating camera stream on port '%s' for camera '%s'" %
                             (stream_port, camera_name))

            self.add_instance(camera_name, CameraInstance(
                process_function=get_sender_function(camera_config.get_source_type()),
                camera=camera,
                stream_port=stream_port,
                hostname=self.hostname
            ))

        self.start_instance(camera_name)

        return self.get_instance(camera_name).get_stream_address()

    def set_camera_instance_config(self, camera_name, new_config):
        self.config_manager.save_camera_config(camera_name, new_config)

        if not self.is_instance_present(camera_name):
            return

        camera_instance = self.get_instance(camera_name)
        camera_instance.set_parameter(new_config)

    def save_script(self, script_name, script):
        return self.user_scripts_manager.save_script(script_name, script)

    def delete_script(self, script_name):
        return self.user_scripts_manager.delete_script(script_name)


class CameraInstance(InstanceWrapper):
    def __init__(self, process_function, camera, stream_port, hostname=None):

        super(CameraInstance, self).__init__(camera.get_name(), process_function, stream_port,
                                             camera, stream_port)

        self.camera = camera

        if not hostname:
            hostname = socket.gethostname()

        if self.get_configuration().get("protocol", "tcp") == "ipc":
            self.stream_address = get_ipc_address(camera.get_name())
        else:
            self.stream_address = "tcp://%s:%d" % (hostname, stream_port)


    def get_info(self):
        return {"stream_address": self.stream_address,
                "is_stream_active": self.is_running(),
                "camera_geometry": self.camera.get_geometry(),
                "camera_name": self.camera.get_name(),
                "last_start_time": self.last_start_time,
                "statistics": self.get_statistics(),
                "config": self.get_configuration()
                }

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
