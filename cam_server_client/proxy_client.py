import logging
import requests
from cam_server_client import config
from cam_server_client.client import Client


_logger = logging.getLogger(__name__)


class ProxyClient(Client):
    def __init__(self, address):
        """
        :param address: Address of the cam API, e.g. http://localhost:10000
        """
        Client.__init__(self, address, config.PROXY_REST_INTERFACE_PREFIX, None)

    def get_servers_info(self):
        """
        Return the info of the server pool of the proxy server.
        For administrative purposes only.
        :return: Dictionary  server -> load, instances
        """
        rest_endpoint = "/servers"
        server_response = requests.get(self.api_address_format % rest_endpoint).json()

        self.validate_response(server_response)
        ret = {}
        servers = server_response["servers"]
        for i in range (len(servers)):
            ret[servers[i]] = {}
            for k in "version", "load", "instances", "cpu", "memory", "tx", "rx":
                ret[servers[i]][k] = server_response[k][i]
        return ret

    def get_status_info(self):
        """
        Return instances foer each server in the server pool .
        For administrative purposes only.
        :return: Dictionary server -> instances
        """
        rest_endpoint = "/status"
        server_response = requests.get(self.api_address_format % rest_endpoint).json()

        return self.validate_response(server_response)["servers"]


    def get_instances_info(self):
        """
        Return the info of all instances in the server pool .
        For administrative purposes only.
        :return: Dictionary
        """
        return self.get_server_info()["active_instances"]

    def get_config(self):
        """
        Return the proxy configuration.
        :return: Proxy configuration.
        """
        rest_endpoint = "/config"

        server_response = requests.get(self.api_address_format % rest_endpoint).json()
        return self.validate_response(server_response)["config"]

    def set_config(self, configuration):
        """
        Set proxy configuration.
        :param configuration: Config to set, in dictionary format.
        :return: Actual applied config.
        """
        rest_endpoint = "/config"

        server_response = requests.post(self.api_address_format % rest_endpoint, json=configuration).json()
        return self.validate_response(server_response)["config"]

    def get_permanent_instances(self):
        """
        Return the permanent instances
        :return: List of string
        """
        rest_endpoint = "/permanent"

        server_response = requests.get(self.api_address_format % rest_endpoint).json()
        return self.validate_response(server_response)["permanent_instances"]

    def set_permanent_instances(self, permanent_instances):
        """
        Set proxy configuration.
        :param configuration: List of string, instance names
        :return: List of string
        """
        rest_endpoint = "/permanent"

        server_response = requests.post(self.api_address_format % rest_endpoint, json=permanent_instances).json()
        return self.validate_response(server_response)["permanent_instances"]

    def get_server_logs(self, server_index_or_name, txt=False):
        """
        Set proxy configuration.
        :param configuration: List of string, instance names
        :return: List of string
        """
        if txt:
            return requests.get(self.address.rstrip("/") + config.API_PREFIX + "/server" + config.LOGS_INTERFACE_PREFIX + "/" + str(server_index_or_name) + "/txt").text
        else:
            return self.validate_response(requests.get(self.address.rstrip("/") + config.API_PREFIX + "/server" + config.LOGS_INTERFACE_PREFIX + "/" + str(server_index_or_name)).json())["logs"]


    def get_instance_logs(self, server_index_or_name, instance_name, txt=False):
        """
        Return the logs.
        :param txt: If True return as text, otherwise as a list
        :return: Version.
        """
        if txt:
            return requests.get(self.address.rstrip("/") + config.API_PREFIX + "/server/instance" + config.LOGS_INTERFACE_PREFIX + "/" + str(server_index_or_name) +  "/" + instance_name + "/txt").text
        else:
            return self.validate_response(requests.get(self.address.rstrip("/") + config.API_PREFIX + "/server/instance" + config.LOGS_INTERFACE_PREFIX + "/" + str(server_index_or_name) +  "/" + instance_name).json())["logs"]