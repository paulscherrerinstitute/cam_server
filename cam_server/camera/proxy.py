import logging
from cam_server.instance_management.proxy import ProxyBase

_logger = logging.getLogger(__name__)


class Proxy(ProxyBase):
    def __init__(self, config_manager, server_pool):
        ProxyBase.__init__(self, server_pool)
        self.config_manager = config_manager

    def get_camera_list(self):
        return self.get_server().get_cameras()

    def set_camera_instance_config(self, camera_name, new_config):
        server = self.get_server(camera_name)
        try:
            server.set_camera_config(camera_name, new_config)
        except:
            # If cannot update config through the server, or if the server is null, then updates directly to disk
            try:
                self.config_manager.save_camera_config(camera_name, new_config)
            except:
                pass

    # TODO: get_camera_image and get_camera_image_bytes connect diretly to the camera.
    #      What should do if there is a connected server to that camera?


