import requests

from cam_server import config


def validate_response(server_response):
    if server_response["state"] != "ok":
        raise ValueError(server_response.get("status", "Unknown error occurred."))

    return server_response


class CamClient(object):
    def __init__(self, address="http://sf-daqsync-01:8888/"):
        """
        :param address: Address of the cam API, e.g. http://localhost:10000
        """

        self.api_address_format = address.rstrip("/") + config.API_PREFIX + config.CAMERA_REST_INTERFACE_PREFIX + "%s"
        self.address = address

    def get_address(self):
        """
        Return the REST api endpoint address.
        """
        return self.address

    def get_server_info(self):
        """
        Return the info of the cam server instance.
        For administrative purposes only.
        :return: Status of the server
        """
        rest_endpoint = "/info"
        server_response = requests.get(self.api_address_format % rest_endpoint).json()

        return validate_response(server_response)["info"]

    def get_cameras(self):
        """
        List existing cameras.
        :return: Currently existing cameras.
        """
        rest_endpoint = ""

        server_response = requests.get(self.api_address_format % rest_endpoint).json()
        return validate_response(server_response)["cameras"]

    def get_camera_config(self, camera_name):
        """
        Return the cam configuration.
        :param camera_name: Name of the cam.
        :return: Camera configuration.
        """
        rest_endpoint = "/%s/config" % camera_name

        server_response = requests.get(self.api_address_format % rest_endpoint).json()
        return validate_response(server_response)["config"]

    def set_camera_config(self, camera_name, configuration):
        """
        Set config on camera.
        :param camera_name: Camera to set the config to.
        :param configuration: Config to set, in dictionary format.
        :return: Actual applied config.
        """
        rest_endpoint = "/%s/config" % camera_name

        server_response = requests.post(self.api_address_format % rest_endpoint, json=configuration).json()
        return validate_response(server_response)["config"]

    def delete_camera_config(self, camera_name):
        """
        Delete config of camera.
        :param camera_name: Camera to set the config to.
        :return: Actual applied config.
        """
        rest_endpoint = "/%s/config" % camera_name

        server_response = requests.delete(self.api_address_format % rest_endpoint).json()
        validate_response(server_response)

    def get_camera_geometry(self, camera_name):
        """
        Get cam geometry.
        :param camera_name: Name of the cam.
        :return: Camera geometry.
        """
        rest_endpoint = "/%s/geometry" % camera_name

        server_response = requests.get(self.api_address_format % rest_endpoint).json()
        return validate_response(server_response)["geometry"]

    def is_camera_online(self, camera_name):
        """
        Return True of camera is online. False otherwise.
        :param camera_name: Name of the cam.
        :return: Camera status.
        """
        rest_endpoint = "/%s/is_online" % camera_name

        server_response = requests.get(self.api_address_format % rest_endpoint).json()
        return validate_response(server_response)["online"]

    def get_camera_image(self, camera_name):
        """
        Return the cam image in PNG format.
        :param camera_name: Camera name.
        :return: server_response content (PNG).
        """
        rest_endpoint = "/%s/image" % camera_name

        server_response = requests.get(self.api_address_format % rest_endpoint)
        return server_response

    def get_camera_image_bytes(self, camera_name):
        """
        Return the cam image bytes.
        :param camera_name: Camera name.
        :return: JSON with bytes and metadata.
        """
        rest_endpoint = "/%s/image_bytes" % camera_name

        server_response = requests.get(self.api_address_format % rest_endpoint).json()
        return validate_response(server_response)["image"]

    def get_instance_stream(self, camera_name):
        """
        Get the camera stream address.
        :param camera_name: Name of the camera to get the address for.
        :return: Stream address.
        """
        rest_endpoint = "/%s" % camera_name

        server_response = requests.get(self.api_address_format % rest_endpoint).json()
        return validate_response(server_response)["stream"]

    def stop_instance(self, camera_name):
        """
        Stop the camera.
        :param camera_name: Name of the camera to stop.
        :return: Response.
        """
        rest_endpoint = "/%s" % camera_name

        server_response = requests.delete(self.api_address_format % rest_endpoint).json()
        validate_response(server_response)

    def stop_all_instances(self):
        """
        Stop all the cameras on the server.
        :return: Response.
        """
        rest_endpoint = ""

        server_response = requests.delete(self.api_address_format % rest_endpoint).json()
        validate_response(server_response)
