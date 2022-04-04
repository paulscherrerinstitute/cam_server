import logging
from cam_server.instance_management.proxy import ProxyBase
from cam_server.camera.source.utils import is_builtin_source
from cam_server import CamClient
import cam_server.camera.source.utils as utils
_logger = logging.getLogger(__name__)


class Manager(ProxyBase):
    def __init__(self, config_manager, user_scripts_manager, config_str, client_timeout=None, update_timeout=None):
        ProxyBase.__init__(self, config_manager, config_str,CamClient, client_timeout, update_timeout)
        self.user_scripts_manager = user_scripts_manager
        utils._user_scripts_manager = user_scripts_manager

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

            source_type = config.get("source_type")
            if source_type == "script":
                source_class = str(config.get("class"))
                if self.user_scripts_manager.exists(source_class):
                    server.set_user_script(source_class, self.user_scripts_manager.get_script(source_class))

    def start_permanent_instance(self, camera, name):
        _logger.info("Starting permanent instance of %s" % (camera))

        self.get_instance_stream(camera)
        server = self.get_server(camera)
        if server is not None:
            cfg = server.get_camera_config(camera)
            cfg["no_client_timeout"] = 0
            server.set_camera_config(camera, cfg)

    def save_script(self, script_name, script):
        if script_name and script:
            self.user_scripts_manager.save_script(script_name, script)
            for server in self.server_pool:
                try:
                    server.set_user_script(script_name, script)
                except:
                    _logger.error("Error setting user script %s on %s" % (script_name, server.get_address()))

    def delete_script(self, script_name):
        if script_name:
            self.user_scripts_manager.delete_script(script_name)
            for server in self.server_pool:
                try:
                    server.delete_script(script_name)
                except:
                    _logger.error("Error deleting user script %s on %s" % (script_name, server.get_address()))
