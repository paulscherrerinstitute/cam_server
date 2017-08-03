import requests

from cam_server import config


class PipelineClient(object):
    def __init__(self, address="http://0.0.0.0:8888/"):
        """
        :param address: Address of the pipeline API, e.g. http://localhost:10000
        """

        self.api_address_format = address.rstrip("/") + config.API_PREFIX + config.PIPELINE_REST_INTERFACE_PREFIX + "%s"

    def get_server_info(self):
        """
        Return the info of the cam server instance.
        For administrative purposes only.
        :return: Status of the server
        """
        rest_endpoint = "/info"
        return requests.get(self.api_address_format % rest_endpoint).json()

    def get_pipelines(self):
        """
        List existing pipelines.
        :return: Currently existing cameras.
        """
        rest_endpoint = ""
        return requests.get(self.api_address_format % rest_endpoint).json()["pipelines"]

    def get_pipeline_config(self, pipeline_name):
        """
        Return the pipeline configuration.
        :param pipeline_name: Name of the pipeline.
        :return: Pipeline configuration.
        """
        rest_endpoint = "/%s/config" % pipeline_name
        return requests.get(self.api_address_format % rest_endpoint).json()

    def get_instance_config(self, instance_id):
        """
        Return the instance configuration.
        :param instance_id: Id of the instance.
        :return: Pipeline configuration.
        """
        rest_endpoint = "/instance/%s/config" % instance_id
        return requests.get(self.api_address_format % rest_endpoint).json()

    def set_pipeline_config(self, pipeline_name, configuration):
        """
        Set config of the pipeline.
        :param pipeline_name: Pipeline to save the config for.
        :param configuration: Config to save, in dictionary format.
        :return: Actual applied config.
        """
        rest_endpoint = "/%s/config" % pipeline_name
        return requests.post(self.api_address_format % rest_endpoint, json=configuration).json()

    def set_instance_config(self, instance_id, configuration):
        """
        Set config of the instance.
        :param instance_id: Instance to apply the config for.
        :param configuration: Config to apply, in dictionary format.
        :return: Actual applied config.
        """
        rest_endpoint = "/instance/%s/config" % instance_id
        return requests.post(self.api_address_format % rest_endpoint, json=configuration).json()

    def stop_instance(self, pipeline_id):
        """
        Stop the pipeline.
        :param pipeline_id: Name of the pipeline to stop.
        :return: Response.
        """
        rest_endpoint = "/%s" % pipeline_id
        return requests.delete(self.api_address_format % rest_endpoint).json()

    def stop_all_instances(self):
        """
        Stop all the pipelines on the server.
        :return: Response.
        """
        rest_endpoint = ""
        return requests.delete(self.api_address_format % rest_endpoint).json()

