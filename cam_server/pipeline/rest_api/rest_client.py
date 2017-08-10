import requests

from cam_server import config


def validate_response(server_response):
    if server_response["state"] != "ok":
        raise ValueError(server_response.get("status", "Unknown error occurred."))

    return server_response


class PipelineClient(object):
    def __init__(self, address="http://0.0.0.0:8889/"):
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
        server_response = requests.get(self.api_address_format % rest_endpoint).json()

        return validate_response(server_response)["info"]

    def get_pipelines(self):
        """
        List existing pipelines.
        :return: Currently existing cameras.
        """
        rest_endpoint = ""
        server_response = requests.get(self.api_address_format % rest_endpoint).json()

        return validate_response(server_response)["pipelines"]

    def get_pipeline_config(self, pipeline_name):
        """
        Return the pipeline configuration.
        :param pipeline_name: Name of the pipeline.
        :return: Pipeline configuration.
        """
        rest_endpoint = "/%s/config" % pipeline_name
        server_response = requests.get(self.api_address_format % rest_endpoint).json()

        return validate_response(server_response)["config"]

    def get_instance_config(self, instance_id):
        """
        Return the instance configuration.
        :param instance_id: Id of the instance.
        :return: Pipeline configuration.
        """
        rest_endpoint = "/instance/%s/config" % instance_id
        server_response = requests.get(self.api_address_format % rest_endpoint).json()

        return validate_response(server_response)["config"]

    def get_instance_info(self, instance_id):
        """
        Return the instance info.
        :param instance_id: Id of the instance.
        :return: Pipeline instance info.
        """
        rest_endpoint = "/instance/%s/info" % instance_id
        server_response = requests.get(self.api_address_format % rest_endpoint).json()

        return validate_response(server_response)["info"]

    def get_instance_stream(self, instance_id):
        """
        Return the instance stream. If the instance does not exist, it will be created.
        Instance will be read only - no config changes will be allowed.
        :param instance_id: Id of the instance.
        :return: Pipeline instance stream.
        """
        rest_endpoint = "/instance/%s" % instance_id
        server_response = requests.get(self.api_address_format % rest_endpoint).json()

        return validate_response(server_response)["stream"]

    def create_instance_from_name(self, pipeline_name, instance_id=None):
        """
        Create a pipeline from a config file. Pipeline config can be changed.
        :param pipeline_name: Name of the pipeline to create.
        :param instance_id: User specified instance id. GUID used if not specified.
        :return: Pipeline instance stream.
        """
        rest_endpoint = "/%s" % pipeline_name

        if instance_id:
            params = {"instance_id": instance_id}
        else:
            params = None

        server_response = requests.post(self.api_address_format % rest_endpoint,
                                        params=params).json()

        validate_response(server_response)

        return server_response["instance_id"], server_response["stream"]

    def create_instance_from_config(self, configuration, instance_id=None):
        """
        Create a pipeline from the provided config. Pipeline config can be changed.
        :param configuration: Config to use with the pipeline.
        :param instance_id: User specified instance id. GUID used if not specified.
        :return: Pipeline instance stream.
        """
        rest_endpoint = ""

        params = None
        if instance_id:
            params = {"instance_id": instance_id}

        server_response = requests.post(self.api_address_format % rest_endpoint,
                                        json=configuration,
                                        params=params).json()

        validate_response(server_response)

        return server_response["instance_id"], server_response["stream"]

    def save_pipeline_config(self, pipeline_name, configuration):
        """
        Set config of the pipeline.
        :param pipeline_name: Pipeline to save the config for.
        :param configuration: Config to save, in dictionary format.
        :return: Actual applied config.
        """
        rest_endpoint = "/%s/config" % pipeline_name
        server_response = requests.post(self.api_address_format % rest_endpoint, json=configuration).json()

        return validate_response(server_response)["config"]

    def delete_pipeline_config(self, pipeline_name):
        """
        Delete a pipeline config.
        :param pipeline_name: Name of pipeline config to delete.
        """
        rest_endpoint = "/%s/config" % pipeline_name

        server_response = requests.delete(self.api_address_format % rest_endpoint).json()
        validate_response(server_response)

    def set_instance_config(self, instance_id, configuration):
        """
        Set config of the instance.
        :param instance_id: Instance to apply the config for.
        :param configuration: Config to apply, in dictionary format.
        :return: Actual applied config.
        """
        rest_endpoint = "/instance/%s/config" % instance_id
        server_response = requests.post(self.api_address_format % rest_endpoint, json=configuration).json()

        return validate_response(server_response)["config"]

    def stop_instance(self, instance_id):
        """
        Stop the pipeline.
        :param instance_id: Name of the pipeline to stop.
        """
        rest_endpoint = "/%s" % instance_id
        server_response = requests.delete(self.api_address_format % rest_endpoint).json()

        validate_response(server_response)

    def stop_all_instances(self):
        """
        Stop all the pipelines on the server.
        """
        rest_endpoint = ""
        server_response = requests.delete(self.api_address_format % rest_endpoint).json()

        validate_response(server_response)

    def collect_background(self, camera_name, n_images=None):
        """
        Collect the background image on the selected camera.
        :param camera_name: Name of the camera to collect the background on.
        :param n_images: Number of images to collect the background on.
        :return: Background id.
        """
        params = None
        if n_images:
            params = {"n_images": n_images}

        rest_endpoint = "/camera/%s/background" % camera_name
        server_response = requests.post(self.api_address_format % rest_endpoint, params=params).json()

        return validate_response(server_response)["background_id"]
