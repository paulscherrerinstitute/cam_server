import os
import time

import requests

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


    def get_instance_logs(self, instance_name, txt=False):
        """
        Return the logs.
        :param txt: If True return as text, otherwise as a list
        :return: Version.
        """
        if txt:
            return requests.get(self.address.rstrip("/") + config.API_PREFIX + config.LOGS_INTERFACE_PREFIX + "/instance/" + instance_name  + "/txt").text
        else:
            return self.validate_response(requests.get(self.address.rstrip("/") + config.API_PREFIX + config.LOGS_INTERFACE_PREFIX + "/instance/" + instance_name).json())["logs"]



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

    def delete_instance(self, instance_id):
        """
        Stop and deletes the instance.
        :param instance_id: Name of the instance to stop.
        """
        rest_endpoint = "/%s/del" % instance_id
        server_response = requests.delete(self.api_address_format % rest_endpoint, timeout=self.timeout).json()
        self.validate_response(server_response)


    def set_config(self, name, configuration):
        """
        Set config of the pipeline.
        :param name: name of the config to save.
        :param configuration: Config to save, in dictionary format.
        :return: Actual applied config.
        """
        rest_endpoint = "/%s/config" % name
        server_response = requests.post(self.api_address_format % rest_endpoint, json=configuration,
                                        timeout=self.timeout).json()

        return self.validate_response(server_response)["config"]

    def delete_config(self, name):
        """
        Delete a pipeline config.
        :param pipeline_name: Name of config to delete.
        """
        rest_endpoint = "/%s/config" % name

        server_response = requests.delete(self.api_address_format % rest_endpoint, timeout=self.timeout).json()
        self.validate_response(server_response)

    def get_config(self, name):
        """
        Return the  configuration.
        :param pipeline_name: Name of the config.
        :return: Pipeline configuration.
        """
        rest_endpoint = "/%s/config" % name
        server_response = requests.get(self.api_address_format % rest_endpoint, timeout=self.timeout).json()

        return self.validate_response(server_response)["config"]

    def get_config_names(self):
        """
        List existing configuration2.
        :return:
        """
        rest_endpoint = "/config_names"
        server_response = requests.get(self.api_address_format % rest_endpoint, timeout=self.timeout).json()

        return self.validate_response(server_response)["config_names"]


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
        self.validate_response(server_response)

    def get_user_script(self, script_name):
        """
        Read user script file bytes.
        :param filename: Script name on the server.
        :return: file bytes
        """
        rest_endpoint = "/script/%s/script_bytes" % script_name
        server_response = requests.get(self.api_address_format % rest_endpoint, timeout=self.timeout).json()
        return self.validate_response(server_response)["script"]

    def delete_script(self, script_name):
        """
        Delete user script file bytes.
        :param filename: Script name on the server.
        """
        rest_endpoint = "/script/%s/script_bytes" % script_name
        server_response = requests.delete(self.api_address_format % rest_endpoint, timeout=self.timeout).json()
        self.validate_response(server_response)

    def upload_user_script(self, filename):
        """
        Upload user script file.
        :param filename: Local script file name.
        :return:
        """
        script_name = os.path.basename(filename)

        mode = "rb" if script_name.endswith(".so") else "r"
        with open(filename, mode) as data_file:
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

        mode = "wb" if script_name.endswith(".so") else "w"
        with open(filename, mode) as data_file:
            data_file.write(script)
        return filename

    def get_user_scripts(self):
        """
        List user scripts.
        :return: List of names of scripts on server
        """
        rest_endpoint = "/script"
        server_response = requests.get(self.api_address_format % rest_endpoint, timeout=self.timeout).json()
        return self.validate_response(server_response)["scripts"]

