import logging
from cam_server.instance_management.proxy import ProxyBase

_logger = logging.getLogger(__name__)
from cam_server import PipelineClient

class Proxy(ProxyBase):
    def __init__(self, config_manager, background_manager, cam_server_client, config_str, server_timeout=None):
        ProxyBase.__init__(self, config_manager, config_str, PipelineClient, server_timeout)
        self.background_manager = background_manager
        self.cam_server_client = cam_server_client

    def get_pipeline_list(self):
        return self.get_server().get_pipelines()

    def create_pipeline(self, pipeline_name=None, configuration=None, instance_id=None):
        if pipeline_name is not None:
            instance_id, stream_address = self.get_server().create_instance_from_name(pipeline_name, instance_id,
                                                                                      configuration)
        elif configuration is not None:
            instance_id, stream_address = self.get_server().create_instance_from_config(configuration, instance_id)
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


    def get_instance_exit_code(self, instance_name):
        status = self.get_status()
        server = self.get_server(instance_name, status)
        if server is not None:
            return server.get_instance_exit_code(instance_name)

    def get_instance_stream_from_config(self, configuration):
        #TODO
        status = self.get_status()
        #server = self.get_server_for_camera(camera_name, status)
        server = self.get_server()
        if server is None:
            server = self.get_free_server(None, status)
        return server.get_instance_stream_from_config(configuration)

    def update_instance_config(self, instance_name, config_updates):
        server = self.get_server(instance_name)
        if server is not None:
            server.set_instance_config(instance_name, config_updates)

    def save_pipeline_config(self, pipeline_name, config):
        return self.config_manager.save_pipeline_config(pipeline_name, config)

    def collect_background(self, camera_name, number_of_images):
        return self.background_manager.collect_background(self.cam_server_client, camera_name, number_of_images)