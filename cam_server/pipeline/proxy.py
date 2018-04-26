from logging import getLogger

_logger = getLogger(__name__)


class ProxyManager(object):
    def __init__(self):
        pass

    def get_pipeline_list(self):
        return self.config_manager.get_pipeline_list()

    def create_pipeline(self, pipeline_name=None, configuration=None, instance_id=None):
        pass

    def get_instance_stream(self, instance_id):
        pass

    def get_instance_stream_from_config(self, configuration):
        pass

    def get_pipeline_config(self):
        pass

    def update_instance_config(self, instance_id, config_updates):
        pass

    def get_info(self):
        pass

    def stop_instance(self, instance_name):
        pass

    def stop_all_instances(self):
        pass
