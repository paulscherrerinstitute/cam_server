import logging
from cam_server.instance_management.proxy import ProxyBase

_logger = logging.getLogger(__name__)


class Manager(ProxyBase):
    def __init__(self, config_manager, background_manager, cam_server_client, server_pool):
        ProxyBase.__init__(self, server_pool)
        self.config_manager = config_manager
        self.background_manager = background_manager
        self.cam_server_client = cam_server_client

    def get_current_servers_for_camera(self, camera, status=None):
        if not status:
            status = self.get_status()
        ret = []
        for server in status:
            for instance in status[server]:
                if camera == status[server][instance]['camera_name']:
                    ret.append(self.get_server_from_address(server))
                    break
        return ret

    def get_pipeline_list(self):
        return self.config_manager.get_pipeline_list()

    def create_pipeline(self, pipeline_name=None, configuration=None, instance_id=None):
        status = self.get_status()
        server = self.get_server(instance_id, status)
        if server is None:
            server = self.get_free_server(instance_id, status)
        if pipeline_name is not None:
            # Update volatile config
            config = self.config_manager.get_pipeline_config(pipeline_name)
            server.save_pipeline_config(pipeline_name, config)

            self._check_background(server, config)
            instance_id, stream_address = server.create_instance_from_name(pipeline_name, instance_id)
        elif configuration is not None:
            instance_id, stream_address = server.create_instance_from_config(configuration, instance_id)
        else:
            raise Exception("Invalid parameters")
        return instance_id, stream_address

    def get_instance_configuration(self, instance_name):
        server = self.get_server(instance_name)
        if server is not None:
            return server.get_instance_config(instance_name)
        raise ValueError("Instance '%s' does not exist." % instance_name)

    def get_instance_info(self, instance_name):
        server = self.get_server(instance_name)
        if server is not None:
            return server.get_instance_info(instance_name)
        raise ValueError("Instance '%s' does not exist." % instance_name)

    def get_instance_stream_from_config(self, configuration):
        #TODO
        status = self.get_status()
        camera_name = configuration["camera_name"]
        servers = self.get_current_servers_for_camera(camera_name, status)
        if len(servers) > 0:
            server = servers[0]
        else:
            server = self.get_free_server(None, status)

        return server.get_instance_stream_from_config(configuration)

    def update_instance_config(self, instance_name, config_updates):
        server = self.get_server(instance_name)
        if server is not None:
            self._check_background(server, config_updates)
            server.set_instance_config(instance_name, config_updates)

    def collect_background(self, camera_name, number_of_images):
        background_id = self.background_manager.collect_background(self.cam_server_client, camera_name, number_of_images)
        for server in self.get_current_servers_for_camera(camera_name):
            image_array = self.background_manager.get_background(background_id)
            server.set_background_image_bytes(background_id, image_array)
        return background_id

    def _check_background(self, server, config):
        if config.get("image_background_enable"):
            image_background = config.get("image_background")
            if image_background:
                # Check if the background can be loaded
                image_array = self.background_manager.get_background(image_background)
                server.set_background_image_bytes(image_background, image_array)

    def save_pipeline_config(self, pipeline_name, config):
        self.config_manager.save_pipeline_config(pipeline_name, config)
        for server in self.server_pool:
            try:
                server.save_pipeline_config(pipeline_name, config)
            except:
                pass