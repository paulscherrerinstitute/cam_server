import requests
import pickle
import os
import json
import time
from bsread import source, SUB

from cam_server import config
from cam_server.utils import get_host_port_from_stream_address
from cam_server.instance_management.rest_api import validate_response



class PipelineClient(object):
    def __init__(self, address="http://sf-daqsync-01:8889/", timeout = None):
        """
        :param address: Address of the pipeline API, e.g. http://localhost:10000
        """
        self.address = address
        self.api_address_format = address.rstrip("/") + config.API_PREFIX + config.PIPELINE_REST_INTERFACE_PREFIX + "%s"
        self.timeout = timeout

    def get_address(self):
        """
        Return the REST api endpoint address.
        """
        return self.address

    def get_server_info(self, timeout = None):
        """
        Return the info of the cam server instance.
        For administrative purposes only.
        Timeout parameter for managers to update more efficiently.
        :return: Status of the server
        """
        rest_endpoint = "/info"
        server_response = requests.get(self.api_address_format % rest_endpoint, timeout=timeout if timeout else self.timeout).json()

        return validate_response(server_response)["info"]

    def is_instance_running(self, instance_id):
        return instance_id in self.get_server_info()["active_instances"]

    def wait_instance_completed(self, instance_id, timeout=None):
        start = time.time()
        while self.is_instance_running(instance_id):
            if timeout:
                if time.time()-start > timeout:
                    raise TimeoutError()
            time.sleep(0.2)

    def get_pipelines(self):
        """
        List existing pipelines.
        :return: Currently existing cameras.
        """
        rest_endpoint = ""
        server_response = requests.get(self.api_address_format % rest_endpoint, timeout=self.timeout).json()

        return validate_response(server_response)["pipelines"]

    def get_pipeline_config(self, pipeline_name):
        """
        Return the pipeline configuration.
        :param pipeline_name: Name of the pipeline.
        :return: Pipeline configuration.
        """
        rest_endpoint = "/%s/config" % pipeline_name
        server_response = requests.get(self.api_address_format % rest_endpoint, timeout=self.timeout).json()

        return validate_response(server_response)["config"]

    def get_instance_config(self, instance_id):
        """
        Return the instance configuration.
        :param instance_id: Id of the instance.
        :return: Pipeline configuration.
        """
        rest_endpoint = "/instance/%s/config" % instance_id
        server_response = requests.get(self.api_address_format % rest_endpoint, timeout=self.timeout).json()

        return validate_response(server_response)["config"]

    def get_instance_info(self, instance_id):
        """
        Return the instance info.
        :param instance_id: Id of the instance.
        :return: Pipeline instance info.
        """
        rest_endpoint = "/instance/%s/info" % instance_id
        server_response = requests.get(self.api_address_format % rest_endpoint, timeout=self.timeout).json()

        return validate_response(server_response)["info"]

    def get_instance_exit_code(self, instance_id):
        """
        Return the instance exit code.
        :param instance_id: Id of the instance.
        :return: Pipeline exit code.
        """
        rest_endpoint = "/instance/%s/exitcode" % instance_id
        server_response = requests.get(self.api_address_format % rest_endpoint, timeout=self.timeout).json()

        return validate_response(server_response)["exitcode"]

    def get_instance_stream(self, instance_id):
        """
        Return the instance stream. If the instance does not exist, it will be created.
        Instance will be read only - no config changes will be allowed.
        :param instance_id: Id of the instance.
        :return: Pipeline instance stream.
        """
        rest_endpoint = "/instance/%s" % instance_id
        server_response = requests.get(self.api_address_format % rest_endpoint, timeout=self.timeout).json()

        return validate_response(server_response)["stream"]

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
        validate_response(server_response)
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
                                        params=params, timeout=self.timeout).json()

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
        server_response = requests.post(self.api_address_format % rest_endpoint, json=configuration,
                                        timeout=self.timeout).json()

        return validate_response(server_response)["config"]

    def delete_pipeline_config(self, pipeline_name):
        """
        Delete a pipeline config.
        :param pipeline_name: Name of pipeline config to delete.
        """
        rest_endpoint = "/%s/config" % pipeline_name

        server_response = requests.delete(self.api_address_format % rest_endpoint, timeout=self.timeout).json()
        validate_response(server_response)

    def set_instance_config(self, instance_id, configuration):
        """
        Set config of the instance.
        :param instance_id: Instance to apply the config for.
        :param configuration: Config to apply, in dictionary format.
        :return: Actual applied config.
        """
        rest_endpoint = "/instance/%s/config" % instance_id
        server_response = requests.post(self.api_address_format % rest_endpoint, json=configuration, timeout=self.timeout).json()

        return validate_response(server_response)["config"]

    def stop_instance(self, instance_id):
        """
        Stop the pipeline.
        :param instance_id: Name of the pipeline to stop.
        """
        rest_endpoint = "/%s" % instance_id
        server_response = requests.delete(self.api_address_format % rest_endpoint, timeout=self.timeout).json()

        validate_response(server_response)

    def stop_all_instances(self):
        """
        Stop all the pipelines on the server.
        """
        rest_endpoint = ""
        server_response = requests.delete(self.api_address_format % rest_endpoint, timeout=self.timeout).json()

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

    def get_latest_background(self, camera_name):
        """
        Return the latest collected background for a camera.
        :param camera_name: Name of the camera to return the background.
        :return: Background id.
        """

        rest_endpoint = "/camera/%s/background" % camera_name
        server_response = requests.get(self.api_address_format % rest_endpoint, timeout=self.timeout).json()

        return validate_response(server_response)["background_id"]

    def get_cameras(self):
        """
        List available cameras.
        :return: Currently available cameras.
        """
        rest_endpoint = "/camera"

        server_response = requests.get(self.api_address_format % rest_endpoint, timeout=self.timeout).json()
        return validate_response(server_response)["cameras"]

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
        Return the bytes of a a background file.
        :param background_name: Background file name.
        :return: JSON with bytes and metadata.
        """
        rest_endpoint = "/background/%s/image_bytes" % background_name

        server_response = requests.get(self.api_address_format % rest_endpoint, timeout=self.timeout).json()
        return validate_response(server_response)["image"]


    def set_background_image_bytes(self, background_name, image_bytes):
        """
        Return the bytes of a a background file.
        :param background_name: Background file name.
        :return: JSON with bytes and metadata.
        """
        rest_endpoint = "/background/%s/image_bytes" % background_name
        data = pickle.dumps(image_bytes, protocol=0)
        server_response = requests.put(self.api_address_format % rest_endpoint, data=data, timeout=self.timeout).json()
        validate_response(server_response)


    def set_user_script(self, script_name, script_bytes):
        """
        Set user script file on the server.
        :param filename: Script file name
        :param script_bytes: Script contents.
        :return:
        """
        rest_endpoint = "/script/%s/script_bytes" % script_name
        server_response = requests.put(self.api_address_format % rest_endpoint, data=script_bytes,
                                       timeout=self.timeout).json()
        validate_response(server_response)

    def get_user_script(self, script_name):
        """
        Read user script file bytes.
        :param filename: Script name on the server.
        :return: file bytes
        """
        rest_endpoint = "/script/%s/script_bytes" % script_name
        server_response = requests.get(self.api_address_format % rest_endpoint, timeout=self.timeout).json()
        return validate_response(server_response)["script"]

    def upload_user_script(self, filename):
        """
        Upload user script file.
        :param filename: Local script file name.
        :return:
        """
        script_name = os.path.basename(filename)

        with open(filename, "r") as data_file:
            script = data_file.read()
        return self.set_user_script(script_name, script)

    def download_user_script(self, filename):
        """
        Download user script file.
        :param filename: Local script file name.
        :return:
        """
        script_name = os.path.basename(filename)
        script = self.get_user_script(script_name)

        with open(filename, "w") as data_file:
            data_file.write(script)
        return filename

    def get_user_scripts(self):
        """
        List user scripts.
        :return: List of names of scripts on server
        """
        rest_endpoint = "/script"
        server_response = requests.get(self.api_address_format % rest_endpoint, timeout=self.timeout).json()
        return validate_response(server_response)["scripts"]

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

    def get_version(self):
        """
        Return the software version.
        :return: Version.
        """
        rest_endpoint = "/version"

        server_response = requests.get(self.api_address_format % rest_endpoint, timeout=self.timeout).json()
        return validate_response(server_response)["version"]

    def get_logs(self, txt=False):
        """
        Return the logs.
        :param txt: If True return as text, otherwise as a list
        :return: Version.
        """
        if txt:
            return requests.get(self.address.rstrip("/") + config.API_PREFIX + config.LOGS_INTERFACE_PREFIX + "/txt").text
        else:
            return validate_response(requests.get(self.address.rstrip("/") + config.API_PREFIX + config.LOGS_INTERFACE_PREFIX).json())["logs"]
