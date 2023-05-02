import requests
import pickle
import os
import json
import time
import base64
import numpy
from bsread import source, SUB

from cam_server_client import config
from cam_server_client.utils import get_host_port_from_stream_address
from cam_server_client.client import InstanceManagementClient



class PipelineClient(InstanceManagementClient):
    def __init__(self, address="http://sf-daqsync-01:8889/", timeout = None):
        """
        :param address: Address of the pipeline API, e.g. http://localhost:10000
        """
        InstanceManagementClient.__init__(self, address, config.PIPELINE_REST_INTERFACE_PREFIX, None)


    def get_pipelines(self):
        """
        List existing pipelines.
        :return: Currently existing cameras.
        """
        rest_endpoint = ""
        server_response = requests.get(self.api_address_format % rest_endpoint, timeout=self.timeout).json()

        return self.validate_response(server_response)["pipelines"]

    def get_pipeline_config(self, pipeline_name):
        """
        Return the pipeline configuration.
        :param pipeline_name: Name of the pipeline.
        :return: Pipeline configuration.
        """
        return self.get_config(pipeline_name)

    def get_pipeline_groups(self):
        """
        Pipeline groups.
        :return: Dicionary group name ->list of pipelines.
        """
        rest_endpoint = "/groups"
        server_response = requests.get(self.api_address_format % rest_endpoint, timeout=self.timeout).json()
        return self.validate_response(server_response)["groups"]

    def get_instance_config(self, instance_id):
        """
        Return the instance configuration.
        :param instance_id: Id of the instance.
        :return: Pipeline configuration.
        """
        rest_endpoint = "/instance/%s/config" % instance_id
        server_response = requests.get(self.api_address_format % rest_endpoint, timeout=self.timeout).json()

        return self.validate_response(server_response)["config"]

    def get_instance_info(self, instance_id):
        """
        Return the instance info.
        :param instance_id: Id of the instance.
        :return: Pipeline instance info.
        """
        rest_endpoint = "/instance/%s/info" % instance_id
        server_response = requests.get(self.api_address_format % rest_endpoint, timeout=self.timeout).json()

        return self.validate_response(server_response)["info"]

    def get_instance_exit_code(self, instance_id):
        """
        Return the instance exit code.
        :param instance_id: Id of the instance.
        :return: Pipeline exit code.
        """
        rest_endpoint = "/instance/%s/exitcode" % instance_id
        server_response = requests.get(self.api_address_format % rest_endpoint, timeout=self.timeout).json()

        return self.validate_response(server_response)["exitcode"]

    def get_instance_stream(self, instance_id):
        """
        Return the instance stream. If the instance does not exist, it will be created.
        Instance will be read only - no config changes will be allowed.
        :param instance_id: Id of the instance.
        :return: Pipeline instance stream.
        """
        rest_endpoint = "/instance/%s" % instance_id
        server_response = requests.get(self.api_address_format % rest_endpoint, timeout=self.timeout).json()

        return self.validate_response(server_response)["stream"]

    def get_instance_stream_from_config(self, configuration):
        """
        Return an instance stream with the matching configuration.
        If the instance does not exist, it will be created.
        Instance will be read only - no config changes will be allowed.
        :param configuration: Configuration of the instance.
        :return: Pipeline instance stream.
        """
        rest_endpoint = "/instance/"
        server_response = requests.post(self.api_address_format % rest_endpoint,
                                       json=configuration, timeout=self.timeout).json()
        self.validate_response(server_response)
        return server_response["instance_id"], server_response["stream"]

    def create_instance_from_name(self, pipeline_name, instance_id=None, additional_config = None):
        """
        Create a pipeline from a config file. Pipeline config can be changed.
        :param pipeline_name: Name of the pipeline to create.
        :param instance_id: User specified instance id. GUID used if not specified.
        :param additional_config: Optional additional configuration elements
        :return: Pipeline instance stream.
        """
        rest_endpoint = "/%s" % pipeline_name

        if instance_id or additional_config:
            params = {}
            if instance_id:
                params["instance_id"] = instance_id
            if additional_config:
                params["additional_config"] = json.dumps(additional_config)
        else:
            params = None

        server_response = requests.post(self.api_address_format % rest_endpoint,
                                        params=params, timeout=self.timeout).json()

        self.validate_response(server_response)

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
                                        params=params, timeout=self.timeout).json()

        self.validate_response(server_response)

        return server_response["instance_id"], server_response["stream"]

    def save_pipeline_config(self, pipeline_name, configuration):
        """
        Set config of the pipeline.
        :param pipeline_name: Pipeline to save the config for.
        :param configuration: Config to save, in dictionary format.
        :return: Actual applied config.
        """
        return self.set_config(pipeline_name, configuration)

    def delete_pipeline_config(self, pipeline_name):
        """
        Delete a pipeline config.
        :param pipeline_name: Name of pipeline config to delete.
        """
        return self.delete_config(pipeline_name)

    def set_instance_config(self, instance_id, configuration):
        """
        Set config of the instance.
        :param instance_id: Instance to apply the config for.
        :param configuration: Config to apply, in dictionary format.
        :return: Actual applied config.
        """
        rest_endpoint = "/instance/%s/config" % instance_id
        server_response = requests.post(self.api_address_format % rest_endpoint, json=configuration, timeout=self.timeout).json()

        return self.validate_response(server_response)["config"]

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

        return self.validate_response(server_response)["background_id"]

    def get_latest_background(self, camera_name):
        """
        Return the latest collected background for a camera.
        :param camera_name: Name of the camera to return the background.
        :return: Background id.
        """

        rest_endpoint = "/camera/%s/background" % camera_name
        server_response = requests.get(self.api_address_format % rest_endpoint, timeout=self.timeout).json()

        return self.validate_response(server_response)["background_id"]

    def get_backgrounds(self, camera_name):
        """
        Return all collected background for a camera.
        :param camera_name: Name of the camera to return the background.
        :return: List of background ids.
        """

        rest_endpoint = "/camera/%s/backgrounds" % camera_name
        server_response = requests.get(self.api_address_format % rest_endpoint, timeout=self.timeout).json()
        return self.validate_response(server_response)["background_ids"]

    def get_cameras(self):
        """
        List available cameras.
        :return: Currently available cameras.
        """
        rest_endpoint = "/camera"

        server_response = requests.get(self.api_address_format % rest_endpoint, timeout=self.timeout).json()
        return self.validate_response(server_response)["cameras"]

    def get_instance_message(self, instance_id):
        """
        Get a single message from a stream instance.
        :param instance_id: Instance id of the stream.
        :return: Message from the stream.
        """
        instance_address = self.get_instance_stream(instance_id)

        instance_config = self.get_instance_config(instance_id)
        pipeline_type = instance_config["pipeline_type"]

        if pipeline_type != "processing":
            raise ValueError("Cannot get message from '%s' pipeline type." % pipeline_type)

        host, port = get_host_port_from_stream_address(instance_address)

        with source(host=host, port=port, mode=SUB) as stream:
            message = stream.receive()

        return message

    def get_background_image(self, background_name):
        """
        Return a background image in PNG format.
        :param background_name: Background file name.
        :return: server_response content (PNG).
        """
        rest_endpoint = "/background/%s/image" % background_name

        server_response = requests.get(self.api_address_format % rest_endpoint, timeout=self.timeout)
        return server_response

    def get_background_image_bytes(self, background_name):
        """
        Return the bytes of a background file.
        :param background_name: Background file name.
        :return: JSON with bytes and metadata.
        """
        rest_endpoint = "/background/%s/image_bytes" % background_name

        server_response = requests.get(self.api_address_format % rest_endpoint, timeout=self.timeout).json()
        return self.validate_response(server_response)["image"]


    def set_background_image_bytes(self, background_name, image_bytes):
        """
        DEPRECATED: Replaced with set_background_image_array
        """
        self.set_background_image_array(background_name, image_bytes)

    def set_background_image_array(self, background_name, image_array):
        """
        Sets the array of a background file.
        :param background_name: Background file name.
        :param image_array: Numpy array
        :return:
        """
        rest_endpoint = "/background/%s/image_bytes" % background_name
        data = pickle.dumps(image_array, protocol=0)
        server_response = requests.put(self.api_address_format % rest_endpoint, data=data, timeout=self.timeout).json()
        self.validate_response(server_response)

    def get_background_image_array(self, background_name):
        """
        Return a background file array
        :param background_name: Background file name.
        :return: 2d numpy array
        """
        image = self.get_background_image_bytes(background_name)
        dtype = image["dtype"]
        shape = image["shape"]
        bytes = base64.b64decode(image["bytes"].encode())
        return numpy.frombuffer(bytes, dtype=dtype).reshape(shape)

    def set_background(self, filename='', data=None):
        """
        Set the background image. If no arguments are provided, the background image is cleared.

        :param str filename: background image filename. It is used merely to track where
                             the background image is loaded.
        :param ndarray data: background image data.
        """
        rest_endpoint = "/background"

        if data is not None:
            data = data.tolist()

        parameters = {
            "filename": filename,
            "data": data
        }
        server_response = requests.post(self.api_address_format % rest_endpoint, json=parameters).json()
        return self.validate_response(server_response)["state"]


    def set_function_script(self, instance_id, filename):
        """
        Upload user script file and set as pipeline function script for a given instance.
        :param instance_id: Id of the instance.
        :param filename: Script file name.
        :return:
        """
        self.upload_user_script(filename)
        script_name = os.path.basename(filename)
        configuration = {}
        configuration["function"] = script_name
        configuration["reload"] = True
        self.set_instance_config(instance_id, configuration)

        return script_name


    def set_lib(self, lib_name, lib_bytes):
        """
        Set user lib file on the server.
        :param filename: Lib file name
        :param lib_bytes: Lib contents.
        :return:
        """
        rest_endpoint = "/lib/%s/lib_bytes" % lib_name
        server_response = requests.put(self.api_address_format % rest_endpoint, data=lib_bytes,
                                       timeout=self.timeout).json()
        self.validate_response(server_response)

    def get_user_lib(self, lib_name):
        """
        Read user lib file bytes.
        :param filename: Lib name on the server.
        :return: file bytes
        """
        rest_endpoint = "/lib/%s/lib_bytes" % lib_name
        server_response = requests.get(self.api_address_format % rest_endpoint, timeout=self.timeout).json()
        return self.validate_response(server_response)["lib"]

    def delete_lib(self, lib_name):
        """
        Delete user lib file bytes.
        :param filename: Lib name on the server.
        """
        rest_endpoint = "/lib/%s/lib_bytes" % lib_name
        server_response = requests.delete(self.api_address_format % rest_endpoint, timeout=self.timeout).json()
        self.validate_response(server_response)


    def get_libs(self):
        """
        List libs.
        :return: List of names of libs on server
        """
        rest_endpoint = "/lib"
        server_response = requests.get(self.api_address_format % rest_endpoint, timeout=self.timeout).json()
        return self.validate_response(server_response)["libs"]

