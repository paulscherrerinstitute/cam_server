
class CamClient(object):
    def __init__(self, address):
        """
        :param address: Address of the cam API, e.g. http://localhost:10000
        """
        self.api_address_format = address.rstrip("/") + API_PREFIX + "/%s"

    def get_cameras(self):
        """
        List existing cameras.
        :return: Currently existing cameras.
        """
        rest_endpoint = "camera_server"
        return requests.get(self.api_address_format % rest_endpoint).json()

    def get_camera_config(self, camera_name):
        """
        Return the camera_server configuration.
        :param camera_name: Name of the camera_server.
        :return: Camera configuration.
        """
        rest_endpoint = "camera_server/%s" % camera_name
        return requests.get(self.api_address_format % rest_endpoint).json()

    def set_camera_config(self, camera_name, config):
        """
        Set config on camera_server.
        :param camera_name: Camera to set the config to.
        :param config: Config to set, in dictionary format.
        :return: Actual applied config.
        """
        rest_endpoint = "camera_server/%s" % camera_name
        return requests.post(self.api_address_format % rest_endpoint, json=config).json()

    def get_camera_geometry(self, camera_name):
        """
        Get camera_server geometry.
        :param camera_name: Name of the camera_server.
        :return: Camera geometry.
        """
        rest_endpoint = "camera_server/%s/geometry" % camera_name
        return requests.get(self.api_address_format % rest_endpoint).json()

    def get_camera_image(self, camera_name):
        """
        Return the camera_server image in PNG format.
        :param camera_name: Camera name.
        :return: Response content (PNG).
        """
        rest_endpoint = "camera_server/%s/image" % camera_name
        return requests.get(self.api_address_format % rest_endpoint).content

    def create_instance(self, instance):
        """
        Create new instance.
        :param instance: Dictionary with the instance definition.
        :return: Instance definition.
        """
        rest_endpoint = "instance"
        return requests.post(self.api_address_format % rest_endpoint, json=instance).json()

    def get_instance(self, instance_name):
        """
        Get instance definition.
        :param instance_name: Instance name.
        :return: Instance definition.
        """
        rest_endpoint = "instance/%s" % instance_name
        return requests.get(self.api_address_format % rest_endpoint).json()

    def set_instance_config(self, instance_name, config):
        """
        Set the instance config.
        :param instance_name: Instance name.
        :param config: Config to set.
        :return: Instance config.
        """
        rest_endpoint = "instance/%s" % instance_name
        return requests.post(self.api_address_format % rest_endpoint, json=config).json()

    def wait_for_instance(self, instance_name):
        """
        Wait for the instance.
        :param instance_name: Instance to wait for.
        """
        rest_endpoint = "instance/%s/wait" % instance_name
        requests.get(self.api_address_format % rest_endpoint)

    def delete_instance(self, instance_name):
        """
        Delete existing instance.
        :param instance_name: Name of the instance
        """
        rest_endpoint = "instance/%s" % instance_name
        requests.delete(self.api_address_format % rest_endpoint)

    def get_instances(self):
        instance_get_url = self.api_address_format % "instance"
        return requests.get(instance_get_url).json()

    def get_first_stream_address(self):
        """
        Get the first instance stream address.
        :return: ZMQ bs_read resource URI.
        """
        instances_list = self.get_instances()
        if len(instances_list) < 1:
            return None

        instance_data = self.get_instance(instances_list[0])

        return instance_data["stream"]