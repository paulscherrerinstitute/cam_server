import glob
import json
import os
import re

from cam_server import config
from cam_server.camera.receiver import CameraSimulation, Camera


def validate_camera_config(camera_config):
    """
    Verify if the cam_server config has the mandatory attributes.
    :param camera_config:
    :return:
    """
    if not camera_config:
        raise ValueError("Config object cannot be empty.\nConfig: %s" % camera_config)

    if "camera" not in camera_config:
        raise ValueError("'camera' section is mandatory in camera_config.\nConfig: %s" % camera_config)

    if "prefix" not in camera_config["camera"]:
        raise ValueError("'prefix' attribute is mandatory in 'camera' section.\nConfig: %s" % camera_config)


class CameraConfigManager(object):
    def __init__(self, config_provider):
        self.config_provider = config_provider

    def get_camera_list(self):
        """
        Retrieve the list of available cameras.
        :return: List of cameras.
        """
        configured_cameras = self.config_provider.get_available_configs()

        # Add simulation cam_server.
        configured_cameras.append('simulation')

        return configured_cameras

    def get_camera_config(self, camera_name):
        """
        Return the camera configuration.
        :param camera_name: Name of the cam_server to retrieve the config for.
        :return: Camera config dictionary.
        """
        # Simulation cam_server is not defined in the config.
        if camera_name.lower() == 'simulation':
            return {"camera": {"prefix": "simulation"}}

        camera_config = self.config_provider.get_config(camera_name)
        validate_camera_config(camera_config)

        return camera_config

    def load_camera(self, camera_name):
        """
        Load a camera with the given name.
        :param camera_name: Camera to load.
        :return: Camera instance.
        """
        # Simulation cam_server is not defined in the config.
        if camera_name.lower() == 'simulation':
            return CameraSimulation()

        camera_config = self.get_camera_config(camera_name)

        return Camera(**camera_config)

    def save_camera_config(self, camera_name, config_updates):
        """
        Save the camera config changes.
        :param camera_name: Name of the cam_server to save the config to.
        :param config_updates: Config to save.
        """

        if camera_name.lower() == 'simulation':
            raise ValueError("Cannot save config for simulation cam_server.")

        # Get either the existing config, or generate a template one.
        try:
            camera_config = self.config_provider.get_camera_config(camera_name)
        except ValueError:
            # Config does not exist, create an empty template.
            camera_config = {"camera": {"prefix": camera_name}}

        # Check if the update is in a valid format.
        if not isinstance(config_updates.get("camera"), dict):
            raise ValueError("Config update must have a 'camera' dictionary. Provided: %s" % config_updates)

        # Update the config.
        camera_config["camera"].update(config_updates)

        # Validate the new config.
        validate_camera_config(camera_config)

        # Verify if the name and the prefix of the cam_server matches.
        camera_prefix = camera_config["camera"]["prefix"]
        if camera_name != camera_prefix:
            raise ValueError("Provided camera name '%s' does not match the config camera prefix '%s'." %
                             (camera_name, camera_prefix))

        self.config_provider.save_config(camera_config)

    def get_camera_geometry(self, camera_name):
        """
        Returns the camera geometry info.
        :param camera_name: Name of the camera.
        :return: (width, height)
        """
        camera = self.load_camera(camera_name)

        camera.connect()
        width, height = camera.get_geometry()
        camera.disconnect()

        return width, height


class CameraConfigFileStorage(object):
    def __init__(self, config_folder=None):
        """
        Initialize the file config provider.
        :param config_folder: Config folder to search for camera definition. If None, default from config.py will
        be used.
        """
        if not config_folder:
            config_folder = config.DEFAULT_CAMERA_CONFIG_FOLDER
        self.config_folder = config_folder

    def get_available_configs(self):
        """
        Return all available camera configurations for instance name.
        :return: List of available cam_server configs.
        """
        cameras = []
        for camera in glob.glob(self.config_folder + '/*.json'):
            # filter out _parameters.json and _background.json files
            if not (re.match(r'.*_parameters.json$', camera) or re.match(r'.*_background.json$', camera)):
                camera = re.sub(r'.*/', '', camera)
                camera = re.sub(r'.json', '', camera)
                cameras.append(camera)

        return cameras

    def _get_config_filename(self, camera_name):
        """
        Construct the filename of the camera config.
        :param camera_name: Camera name.
        :return:
        """
        return self.config_folder + '/' + camera_name + '.json'

    def get_config(self, camera_name):
        """
        Return config for a camera.
        :param camera_name: Camera config to retrieve.
        :return: Dict containing the camera config.
        """

        config_file = self._get_config_filename(camera_name)

        # The config file does not exist
        if not os.path.isfile(config_file):
            raise ValueError("Unable to load camera '%s'. Config '%s' does not exist." %
                             (camera_name, config_file))

        with open(config_file) as data_file:
            configuration = json.load(data_file)

        return configuration

    def save_config(self, camera_name, camera_config):
        """
        Update an existing camera config.
        :param camera_name: Name of the camera to same the config for.
        :param camera_config: Configuration to persist.
        """
        target_config_file = self._get_config_filename(camera_name)

        with open(target_config_file, 'w') as data_file:
            json.dump(camera_config, data_file, indent=True)