import requests

from cam_server import config


class CamClient(object):
    def __init__(self, address="http://0.0.0.0:8888/"):
        """
        :param address: Address of the cam API, e.g. http://localhost:10000
        """

        self.api_address_format = address.rstrip("/") + config.API_PREFIX + "/%s"

    def get_server_info(self):
        """
        Return the info of the cam server instance.
        For administrative purposes only.
        :return: Status of the server
        """
        rest_endpoint = "cam_server/info"
        return requests.get(self.api_address_format % rest_endpoint).json()

    def get_cameras(self):
        """
        List existing cameras.
        :return: Currently existing cameras.
        """
        rest_endpoint = "cam_server"
        return requests.get(self.api_address_format % rest_endpoint).json()

    def get_camera_config(self, camera_name):
        """
        Return the cam_server configuration.
        :param camera_name: Name of the cam_server.
        :return: Camera configuration.
        """
        rest_endpoint = "cam_server/%s/config" % camera_name
        return requests.get(self.api_address_format % rest_endpoint).json()

    def set_camera_config(self, camera_name, config):
        """
        Set config on cam_server.
        :param camera_name: Camera to set the config to.
        :param config: Config to set, in dictionary format.
        :return: Actual applied config.
        """
        rest_endpoint = "cam_server/%s" % camera_name
        return requests.post(self.api_address_format % rest_endpoint, json=config).json()

    def get_camera_geometry(self, camera_name):
        """
        Get cam_server geometry.
        :param camera_name: Name of the cam_server.
        :return: Camera geometry.
        """
        rest_endpoint = "cam_server/%s/geometry" % camera_name
        return requests.get(self.api_address_format % rest_endpoint).json()

    def get_camera_image(self, camera_name):
        """
        Return the cam_server image in PNG format.
        :param camera_name: Camera name.
        :return: Response content (PNG).
        """
        rest_endpoint = "cam_server/%s/image" % camera_name
        return requests.get(self.api_address_format % rest_endpoint).content

    def get_camera_stream(self, camera_name):
        """
        Get the camera stream address.
        :param camera_name: Name of the camera to get the address for.
        :return: Stream address.
        """
        rest_endpoint = "cam_server/%s" % camera_name
        return requests.get(self.api_address_format % rest_endpoint).json()

