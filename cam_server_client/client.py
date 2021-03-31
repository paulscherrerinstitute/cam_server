import requests
import time

from cam_server_client import config


class Client(object):
    def __init__(self, address, prefix, timeout=None):
        """
        :param address: Address of the API, e.g. http://localhost:10000
        """
        self.api_address_format = address.rstrip("/") + config.API_PREFIX + prefix + "%s"
        self.address = address
        self.timeout = timeout

    def validate_response(self, server_response):
        if server_response["state"] != "ok":
            raise ValueError(server_response.get("status", "Unknown error occurred."))
        return server_response


    def get_address(self):
        """
        Return the REST api endpoint address.
        """
        return self.address

    def get_version(self):
        """
        Return the software version.
        :return: Version.
        """
        rest_endpoint = "/version"

        server_response = requests.get(self.api_address_format % rest_endpoint, timeout=self.timeout).json()
        return self.validate_response(server_response)["version"]

    def get_logs(self, txt=False):
        """
        Return the logs.
        :param txt: If True return as text, otherwise as a list
        :return: Version.
        """
        if txt:
            return requests.get(self.address.rstrip("/") + config.API_PREFIX + config.LOGS_INTERFACE_PREFIX + "/txt").text
        else:
            return self.validate_response(requests.get(self.address.rstrip("/") + config.API_PREFIX + config.LOGS_INTERFACE_PREFIX).json())["logs"]

    def get_server_info(self, timeout = None):
        """
        Return the info of the cam server instance.
        For administrative purposes only.
        Timeout parameter for managers to update more efficiently
        :return: Status of the server
        """
        rest_endpoint = "/info"
        server_response = requests.get(self.api_address_format % rest_endpoint, timeout=timeout if timeout else self.timeout).json()

        return self.validate_response(server_response)["info"]


class InstanceManagementClient(Client):
    def __init__(self, address, prefix, timeout = None):
        Client.__init__(self, address, prefix, timeout)

    def is_instance_running(self, instance_id):
        return instance_id in self.get_server_info()["active_instances"]

    def wait_instance_completed(self, instance_id, timeout=None):
        start = time.time()
        while self.is_instance_running(instance_id):
            if timeout:
                if time.time()-start > timeout:
                    raise TimeoutError()
            time.sleep(0.2)

    def stop_instance(self, instance_id):
        """
        Stop the instance.
        :param instance_id: Name of the instance to stop.
        """
        rest_endpoint = "/%s" % instance_id
        server_response = requests.delete(self.api_address_format % rest_endpoint, timeout=self.timeout).json()

        self.validate_response(server_response)

    def stop_all_instances(self):
        """
        Stop all the instances on the server.
        """
        rest_endpoint = ""
        server_response = requests.delete(self.api_address_format % rest_endpoint, timeout=self.timeout).json()

        self.validate_response(server_response)




