import logging
from cam_server.instance_management.proxy import ProxyBase
from cam_server import CamClient

_logger = logging.getLogger(__name__)


class Manager(ProxyBase):
    def __init__(self, config_manager, config_str, client_timeout=None, update_timeout=None):
        ProxyBase.__init__(self, config_manager, config_str,CamClient, client_timeout, update_timeout)

    def get_config_names(self):
        return self.get_camera_list()

    def get_camera_list(self):
        return self.config_manager.get_camera_list()

    def set_camera_instance_config(self, camera_name, new_config):
        self.config_manager.save_camera_config(camera_name, new_config)
        server = self.get_server(camera_name)
        if server is not None:
            server.set_camera_config(camera_name, new_config)

    def on_creating_server_stream(self, server, instance_name, port):
        if (instance_name is not None) and (server is not None):
            # Update volatile config
            config = self.config_manager.get_camera_config(instance_name).get_configuration()
            if port:
                config["port"] = port
            server.set_camera_config(instance_name, config)

    def start_permanent_instance(self, camera, name):
        _logger.info("Starting permanent instance of %s" % (camera))

        self.get_instance_stream(camera)
        server = self.get_server(camera)
        if server is not None:
            cfg = server.get_camera_config(camera)
            cfg["no_client_timeout"] = 0
            server.set_camera_config(camera, cfg)
