import requests

from cam_server_client.client import InstanceManagementClient
from cam_server_client import config


class CamClient(InstanceManagementClient):
    def __init__(self, address="http://sf-daqsync-01:8888/", timeout = None):
        """
        :param address: Address of the cam API, e.g. http://localhost:10000
        """
        InstanceManagementClient.__init__(self, address, config.CAMERA_REST_INTERFACE_PREFIX, None)


    def get_cameras(self):
        """
        List existing cameras.
        :return: Currently existing cameras.
        """
        rest_endpoint = ""

        server_response = requests.get(self.api_address_format % rest_endpoint, timeout=self.timeout).json()
        return self.validate_response(server_response)["cameras"]

    def get_camera_aliases(self):
        """
        Cameras aliases.
        :return: Dicionary alias->name.
        """
        rest_endpoint = "/aliases"

        server_response = requests.get(self.api_address_format % rest_endpoint, timeout=self.timeout).json()
        return self.validate_response(server_response)["aliases"]

    def get_camera_groups(self):
        """
        Cameras groups.
        :return: Dicionary group name ->list of cameras.
        """
        rest_endpoint = "/groups"

        server_response = requests.get(self.api_address_format % rest_endpoint, timeout=self.timeout).json()
        return self.validate_response(server_response)["groups"]

    def get_camera_config(self, camera_name):
        """
        Return the cam configuration.
        :param camera_name: Name of the cam.
        :return: Camera configuration.
        """
        return self.get_config(camera_name)

    def set_camera_config(self, camera_name, configuration):
        """
        Set config on camera.
        :param camera_name: Camera to set the config to.
        :param configuration: Config to set, in dictionary format.
        :return: Actual applied config.
        """
        return self.set_config(camera_name, configuration)

    def delete_camera_config(self, camera_name):
        """
        Delete config of camera.
        :param camera_name: Camera to set the config to.
        :return: Actual applied config.
        """
        return self.delete_config(camera_name)

    def get_camera_geometry(self, camera_name):
        """
        Get cam geometry.
        :param camera_name: Name of the cam.
        :return: Camera geometry.
        """
        rest_endpoint = "/%s/geometry" % camera_name

        server_response = requests.get(self.api_address_format % rest_endpoint, timeout=self.timeout).json()
        return self.validate_response(server_response)["geometry"]

    def is_camera_online(self, camera_name):
        """
        Return True of camera is online. False otherwise.
        :param camera_name: Name of the cam.
        :return: Camera status.
        """
        rest_endpoint = "/%s/is_online" % camera_name

        server_response = requests.get(self.api_address_format % rest_endpoint, timeout=self.timeout).json()
        return self.validate_response(server_response)["online"]

    def get_camera_image(self, camera_name):
        """
        Return the cam image in PNG format.
        :param camera_name: Camera name.
        :return: server_response content (PNG).
        """
        rest_endpoint = "/%s/image" % camera_name

        server_response = requests.get(self.api_address_format % rest_endpoint, timeout=self.timeout)
        return server_response

    def get_camera_image_bytes(self, camera_name):
        """
        Return the cam image bytes.
        :param camera_name: Camera name.
        :return: JSON with bytes and metadata.
        """
        rest_endpoint = "/%s/image_bytes" % camera_name

        server_response = requests.get(self.api_address_format % rest_endpoint, timeout=self.timeout).json()
        return self.validate_response(server_response)["image"]

    def get_instance_stream(self, camera_name):
        """
        Get the camera stream address.
        :param camera_name: Name of the camera to get the address for.
        :return: Stream address.
        """
        rest_endpoint = "/%s" % camera_name

        server_response = requests.get(self.api_address_format % rest_endpoint, timeout=self.timeout).json()
        return self.validate_response(server_response)["stream"]




