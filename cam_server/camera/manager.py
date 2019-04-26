import logging
from cam_server.instance_management.proxy import ProxyBase
from cam_server import CamClient

_logger = logging.getLogger(__name__)


class Manager(ProxyBase):
    def __init__(self, config_manager, configuration):
        server_pool = [CamClient(server) for server in configuration.keys()]
        ProxyBase.__init__(self, server_pool)
        self.config_manager = config_manager

    def get_camera_list(self):
        return self.config_manager.get_camera_list()

    def set_camera_instance_config(self, camera_name, new_config):
        self.config_manager.save_camera_config(camera_name, new_config)
        server = self.get_server(camera_name)
        if server is not None:
            server.set_camera_config(camera_name, new_config)

    def save_pipeline_config(self, pipeline_name, config):
        for server in self.server_pool:
            try:
                info = status[server.get_address()]
                if instance_name in info.keys():
                    return server
            except:
                pass
        return self.config_manager.save_pipeline_config(pipeline_name, config)

    def get_free_server(self, instance_name=None, status=None):
        server = ProxyBase.get_free_server(self,instance_name, status)
        if (instance_name is not None) and (server is not None):
            # Update volatile config
            config = self.config_manager.get_camera_config(instance_name).get_configuration()
            server.set_camera_config(instance_name, config)
        return server
