import glob
import json
import os
import re
import sys
from os.path import basename

from cam_server import config


class ConfigFileStorage(object):
    def __init__(self, config_folder=None):
        """
        Initialize the file config provider.
        :param config_folder: Config folder to search for camera definition. If None, default from config.py will
        be used.
        """

        if len(config_folder) > 1 and config_folder[-1] == '/':
            config_folder = config_folder[:-1]

        if not config_folder:
            config_folder = config.DEFAULT_CAMERA_CONFIG_FOLDER
        self.config_folder = config_folder

    def get_available_configs(self):
        """
        Return all available  configurations .
        :return: List of available configs.
        """
        cameras = []
        for camera in glob.glob(self.config_folder + '/*.json'):
            # filter out _parameters.json and _background.json files
            if not (re.match(r'.*_parameters.json$', camera) or
                    re.match(r'.*_background.json$', camera) or
                    re.match(r'.*/permanent_instances.json$', camera) or
                    re.match(r'.*/servers.json$', camera)):
                camera = re.sub(r'.*/', '', camera)
                camera = re.sub(r'.json', '', camera)
                cameras.append(camera)

        return cameras

    def _get_config_filename(self, config_name):
        """
        Construct the filename of the camera config.
        :param config_name: Config name.
        :return:
        """
        return self.config_folder + '/' + config_name + '.json'

    def _get_named_configuration(self, config_name, as_text=False):
        """
        Load the entire configuration file (which includes also section we might not be interested in).
        :param config_name: Name of the configuration to load.
        :return: Dictionary with the config.
        """
        config_file = self._get_config_filename(config_name)

        # The config file does not exist
        if not os.path.isfile(config_file):
            raise ValueError("Unable to load config '%s'. Config file '%s' does not exist." %
                             (config_name, config_file))

        with open(config_file) as data_file:
            if as_text:
                configuration = data_file.read()
            else:
                configuration = json.load(data_file)

        return configuration

    def get_config(self, config_name, as_text=False):
        """
        Return config for a camera.
        :param config_name: Camera config to retrieve.
        :return: Dict containing the camera config.
        """

        configuration = self._get_named_configuration(config_name, as_text)
        return configuration

    def save_config(self, config_name, configuration):
        """
        Update an existing camera config.
        :param config_name: Name of the config to save.
        :param configuration: Configuration to persist.
        """
        target_config_file = self._get_config_filename(config_name)
        # We need to enforce this for the file storage - retrieve the files by config name.
        configuration["name"] = config_name

        with open(target_config_file, 'w') as data_file:
            if isinstance(configuration, str):
                json.write(configuration)
            else:
                json.dump(configuration, data_file, indent=True)

    def delete_config(self, config_name):
        """
        Delete the provided config.
        :param config_name: Config name to delete.
        """
        target_config_file = self._get_config_filename(config_name)
        os.remove(target_config_file)


class TransientConfig(object):
    def __init__(self, configuration = {}):
        """
        Initialize the transient config provider.
        be used.
        """
        self.configuration = configuration

    def get_available_configs(self):
        """
        Return all available  configurations .
        :return: List of available configs.
        """
        return self.configuration.keys()

    def get_config(self, config_name):
        """
        Return config for a camera.
        :param config_name: Camera config to retrieve.
        :return: Dict containing the camera config.
        """

        if not config_name in self.configuration:
            raise ValueError("Unable to load config '%s'" % (config_name,))
        return self.configuration[config_name]

    def save_config(self, config_name, configuration):
        """
        Update an existing camera config.
        :param config_name: Name of the config to save.
        :param configuration: Configuration to persist.
        """
        self.configuration[config_name]=configuration

    def delete_config(self, config_name):
        """
        Delete the provided config.
        :param config_name: Config name to delete.
        """
        if config_name in self.configuration:
            del self.configuration[config_name]

    """
    def register_rest_interface(self, app):
        from bottle import request
        api_root_address = config.API_PREFIX + config.CAMERA_REST_INTERFACE_PREFIX

        @app.post(api_root_address+ "/configuration")
        def set_configuration():
            self.configuration = request.json

            return {"state": "ok",
                    "status": "Camera configuration saved.",
                    "config": self.configuration}

        @app.get(api_root_address + "is_configured")
        def is_configured(camera_name):
            configured = (self.configuration is not None) and (len(self.configuration) > 0)
            return {"state": "ok",
                    "status": "Is server configured",
                    "configured": configured}
    """


class UserScriptsManager(object):
    def __init__(self, scripts_folder):
        if scripts_folder is not None:
            if len(scripts_folder) > 1 and scripts_folder[-1] == '/':
                scripts_folder = scripts_folder[:-1]

        self.scripts_folder = scripts_folder

        if not os.path.exists(self.get_lib_home()):
            os.makedirs(self.get_lib_home())

    def exists(self, script_name):
        if script_name and self.scripts_folder:
            script_filename = self._get_script_filename(script_name)
            return os.path.isfile(script_filename)
        return False

    def get_home(self):
        return self.scripts_folder

    def get_path(self, script_name):
        if script_name and self.scripts_folder:
            return self._get_script_filename(script_name)
        return None

    def get_file_type(self, script_name):
        try:
            import os
            ext = os.path.splitext(script_name)[1][1:]
            if ext:
                return ext
        except:
            pass
        return None

    def _get_script_filename(self, script_name):
        if not script_name or not self.scripts_folder:
            return None
        file_type = self.get_file_type(script_name)
        if not file_type:
            file_list = glob.glob(self.scripts_folder + "/" + script_name + "*.so")
            lib_name = "" if len(file_list) == 0 else file_list[0]
            lib_exists = os.path.exists(lib_name)

            if os.path.isfile(os.path.join(self.scripts_folder, script_name+".c")):
                script_name += ".c"
            elif os.path.exists(lib_name):
                _, script_name = os.path.split(lib_name)
            else:
                script_name += ".py"
        return os.path.join(self.scripts_folder, script_name)

    def get_script_type(self, script_name):
        if not script_name or not self.scripts_folder:
            return None
        file_type = self.get_file_type(script_name)
        if not file_type:
            file_list = glob.glob(self.scripts_folder + "/" + script_name + "*.so")
            lib_name = "" if len(file_list) == 0 else file_list[0]
            lib_exists = os.path.exists(lib_name)

            if os.path.isfile(os.path.join(self.scripts_folder, script_name + ".c")):
                return "c"
            elif os.path.exists(lib_name):
                return "so"
            return "py"
        return file_type

    def get_script_file_name(self, script_name):
        if not self.get_file_type(script_name):
            ext = self.get_script_type(script_name)
            script_name = script_name + "." + ext
        return script_name

    def get_script(self, script_name):
        script_filename = self._get_script_filename(script_name)
        if script_filename is None:
                return

        if not os.path.isfile(script_filename):
            raise ValueError("Requested script '%s' does not exist." % script_name)

        if self.get_script_type(script_filename) == "so":
            with open(script_filename, "rb") as data_file:
                return data_file.read()


        with open(script_filename, "r") as data_file:
            return data_file.read()

    def save_script(self, script_name, script):

        script_filename = self._get_script_filename(script_name)
        if script_filename is None:
                return

        if self.get_script_type(script_name) == "so":
            with open(script_filename, "wb") as data_file:
                data_file.write(script)
        else:
            if type(script) != str:
                script = script.decode("utf-8")

            with open(script_filename, "w") as data_file:
                data_file.write(script)

    def delete_script(self, script_name):
        """
        Delete the provided script.
        :param script_name: Script name to delete.
        """
        script_filename = self._get_script_filename(script_name)
        if script_filename is None:
                return

        os.remove(script_filename)

    def get_scripts(self):
        if not self.scripts_folder:
            return []
        #scripts = glob.glob(self.scripts_folder + '/*.py')
        scripts = []
        for files in (self.scripts_folder + '/*.py', self.scripts_folder + '/*.c'):
            scripts.extend(glob.glob(files))

        for i in range(len(scripts)):
            scripts[i] = basename(scripts[i])
        return scripts


    def get_lib_home(self):
        if not self.scripts_folder:
            return None

        return self.scripts_folder + "/lib"

    def get_lib_path(self, lib_name):
        if lib_name:
            if os.path.exists(lib_name):
                return lib_name
            lib_home = self.get_lib_home()
            if lib_home:
                return os.path.join(lib_home, lib_name)
        return None


    def get_lib(self, lib_name):
        lib_filename = self.get_lib_path(lib_name)
        if lib_filename is None:
            return

        if not os.path.isfile(lib_filename):
            raise ValueError("Requested lib '%s' does not exist." % lib_name)

        if self.get_file_type(lib_filename) in ["py", "c", "txt", "csv"]:
            with open(lib_filename, "r") as data_file:
                return data_file.read()

        with open(lib_filename, "rb") as data_file:
            return data_file.read()


    def save_lib(self, lib_name, lib):

        lib_filename = self.get_lib_path(lib_name)
        if lib_filename is None:
                return

        if self.get_file_type(lib_filename) in ["py", "c", "txt", "csv"]:
            if type(lib) != str:
                script = lib.decode("utf-8")

            with open(lib_filename, "w") as data_file:
                data_file.write(lib)

        else:
            with open(lib_filename, "wb") as data_file:
                data_file.write(lib)


    def delete_lib(self, lib_name):
        """
        Delete the provided library.
        :param script_name: Library name to delete.
        """
        lib_filename = self.get_lib_path(lib_name)
        if lib_filename is None:
                return
        os.remove(lib_filename)


    def get_libs(self):
        lib_home = self.get_lib_home()
        if not lib_home:
            return []
        libs = []
        for files in (self.lib_home + '/*.*',):
            libs.extend(glob.glob(files))

        for i in range(len(libs)):
            libs[i] = basename(libs[i])
        return libs

    def exists_lib(self, lib):
        lib_filename = self.get_lib_path(lib)
        if lib_filename:
            return os.path.isfile(lib_filename)
        return False
