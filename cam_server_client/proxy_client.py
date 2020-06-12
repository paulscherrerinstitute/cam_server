import logging
import requests
from cam_server_client import config
from cam_server_client.utils import validate_response


_logger = logging.getLogger(__name__)


class ProxyClient(object):
    def __init__(self, address):
        """
        :param address: Address of the cam API, e.g. http://localhost:10000
        """
        self.api_address_format = address.rstrip("/") + config.API_PREFIX + config.PROXY_REST_INTERFACE_PREFIX + "%s"
        self.address = address

    def get_address(self):
        """
        Return the REST api endpoint address.
        """
        return self.address

    def get_servers_info(self):
        """
        Return the info of the server pool of the proxy server.
        For administrative purposes only.
        :return: Dictionary  server -> load, instances
        """
        rest_endpoint = "/servers"
        server_response = requests.get(self.api_address_format % rest_endpoint).json()

        validate_response(server_response)
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

        return validate_response(server_response)["servers"]


    def get_instances_info(self):
        """
        Return the info of all instances in the server pool .
        For administrative purposes only.
        :return: Dictionary
        """
        rest_endpoint = "/info"
        server_response = requests.get(self.api_address_format % rest_endpoint).json()
        return validate_response(server_response)["info"]["active_instances"]

    def get_config(self):
        """
        Return the proxy configuration.
        :return: Proxy configuration.
        """
        rest_endpoint = "/config"

        server_response = requests.get(self.api_address_format % rest_endpoint).json()
        return validate_response(server_response)["config"]

    def set_config(self, configuration):
        """
        Set proxy configuration.
        :param configuration: Config to set, in dictionary format.
        :return: Actual applied config.
        """
        rest_endpoint = "/config"

        server_response = requests.post(self.api_address_format % rest_endpoint, json=configuration).json()
        return validate_response(server_response)["config"]

    def get_version(self):
        """
        Return the software version.
        :return: Version.
        """
        rest_endpoint = "/version"

        server_response = requests.get(self.api_address_format % rest_endpoint).json()
        return validate_response(server_response)["version"]


    def get_permanent_instances(self):
        """
        Return the permanent instances
        :return: List of string
        """
        rest_endpoint = "/permanent"

        server_response = requests.get(self.api_address_format % rest_endpoint).json()
        return validate_response(server_response)["permanent_instances"]

    def set_permanent_instances(self, permanent_instances):
        """
        Set proxy configuration.
        :param configuration: List of string, instance names
        :return: List of string
        """
        rest_endpoint = "/permanent"

        server_response = requests.post(self.api_address_format % rest_endpoint, json=permanent_instances).json()
        return validate_response(server_response)["permanent_instances"]
