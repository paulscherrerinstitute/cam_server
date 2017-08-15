
class LisaInstanceManager:
    def __init__(self, config_manager, background_manager, cam_server_client, hostname=None):
        self.config_manager = config_manager
        self.background_manager = background_manager
        self.cam_server_client = cam_server_client
        self.hostname = hostname

        # self.port_generator = get_port_generator(config.PIPELINE_STREAM_PORT_RANGE)

    def get_pipeline_list(self):
        pass

    def create_pipeline(self, pipeline_name=None, configuration=None, instance_id=None):
        """
        Create the pipeline stream address. Either pass the pipeline name, or the configuration.
        :param pipeline_name: Name of the pipeline to load from config.
        :param configuration: Configuration to load the pipeline with.
        :param instance_id: Name to assign to the instace. It must be unique.
        :return: instance_id, Pipeline stream address.
        """
        pass

    def get_instance_stream(self, instance_id):
        pass

    def update_instance_config(self, instance_id, config_updates):
        pass


