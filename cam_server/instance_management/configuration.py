import glob
import json
import os
import re

from cam_server import config


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