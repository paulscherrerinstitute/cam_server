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
        Return all available  configurations .
        :return: List of available configs.
        """
        cameras = []
        for camera in glob.glob(self.config_folder + '/*.json'):
            # filter out _parameters.json and _background.json files
            if not (re.match(r'.*_parameters.json$', camera) or re.match(r'.*_background.json$', camera)):
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

    def get_config(self, config_name):
        """
        Return config for a camera.
        :param config_name: Camera config to retrieve.
        :return: Dict containing the camera config.
        """

        config_file = self._get_config_filename(config_name)

        # The config file does not exist
        if not os.path.isfile(config_file):
            raise ValueError("Unable to load config '%s'. Config file '%s' does not exist." %
                             (config_name, config_file))

        with open(config_file) as data_file:
            configuration = json.load(data_file)

        return configuration

    def save_config(self, config_name, configuration):
        """
        Update an existing camera config.
        :param config_name: Name of the config to save.
        :param configuration: Configuration to persist.
        """
        target_config_file = self._get_config_filename(config_name)

        with open(target_config_file, 'w') as data_file:
            json.dump(configuration, data_file, indent=True)
