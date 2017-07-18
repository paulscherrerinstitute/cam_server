import glob
import json
import os
import re
from logging import getLogger

from cam_server import config
from cam_server.camera.instance import CameraInstance
from cam_server.camera.sender import process_camera_stream
from cam_server.camera.receiver import CameraSimulation, Camera

_logger = getLogger(__name__)


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


class CameraInstanceManager(object):
    def __init__(self, config_manager):
        self.camera_instances = {}
        self.port_mapping = {}
        self.config_manager = config_manager
        self.port_generator = range(*config.CAMERA_STREAM_PORT_RANGE)

    def _create_camera_instance(self, camera_name):
        """
        Create a new camera instance and add it to the instance pool.
        :param camera_name: Camera name to instantiate.
        """
        stream_port = next(self.port_generator)

        _logger.info("Creating camera instance '%s' on port %d.", (camera_name, stream_port))

        self.camera_instances[camera_name] = CameraInstance(
            process_function=process_camera_stream,
            camera_instance=self.config_manager.load_camera(camera_name),
            stream_port=stream_port
        )

    def get_camera_stream(self, camera_name):
        """
        Get the camera stream address.
        :param camera_name: Name of the camera to get the stream for.
        :return: Camera stream address.
        """

        # Check if the requested camera already exists.
        if camera_name not in self.camera_instances:
            self._create_camera_instance(camera_name)

        camera_instance = self.camera_instances[camera_name]

        # If camera instance is not yet running, start it.
        if not camera_instance.is_running():
            camera_instance.start()
        else:
            # TODO: Signal to the camera to wait for X seconds in case no clients are connected.
            pass

        return camera_instance.stream_address

    def get_info(self):
        """
        Return the instance manager info.
        :return: Dictionary with the info.
        """
        info = {"active_cameras": [camera for camera in self.camera_instances if camera.is_running()]}
        return info

    def stop_camera(self, camera_name):
        """
        Terminate the stream of the specified camera.
        :param camera_name: Name of the camera to stop.
        """
        _logger.info("Stopping camera '%s' instances.", camera_name)

        if camera_name in self.camera_instances:
            self.camera_instances[camera_name].stop()

    def stop_all_cameras(self):
        """
        Terminate the streams of all cameras.
        :return:
        """
        _logger.info("Stopping all camera instances.")

        for camera_name in self.camera_instances:
            self.stop_camera(camera_name)


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
        Return the cam_server configuration.
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
        Load a cam_server with the given name.
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
        Save the cam_server config changes.
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
        camera_prefix = camera_config["cam_server"]["prefix"]
        if camera_name != camera_prefix:
            raise ValueError("Provided cam_server name '%s' does not match the config cam_server prefix '%s'." %
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
        width, height = camera.get_info()
        camera.disconnect()

        return width, height


class CameraConfigFileStorage(object):
    def __init__(self, config_folder=None):
        """
        Initialize the file config provider.
        :param config_folder: Config folder to search for cam_server definition. If None, default from config.py will
        be used.
        """
        if not config_folder:
            config_folder = config.DEFAULT_CAMERA_CONFIG_FOLDER
        self.config_folder = config_folder

    def get_available_configs(self):
        """
        Return all available cam_server configurations for instance name.
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
        Construct the filename of the cam_server config.
        :param camera_name: Camera name.
        :return:
        """
        return self.config_folder + '/' + camera_name + '.json'

    def get_config(self, camera_name):
        """
        Return config for a cam_server.
        :param camera_name: Camera config to retrieve.
        :return: Dict containing the cam_server config.
        """

        config_file = self._get_config_filename(camera_name)

        # The config file does not exist
        if not os.path.isfile(config_file):
            raise ValueError("Unable to load cam_server %s. Config '%s' does not exist." %
                             (camera_name, config_file))

        with open(config_file) as data_file:
            configuration = json.load(data_file)

        return configuration

    def save_config(self, camera_name, camera_config):
        """
        Update an existing cam_server config.
        :param camera_name: Name of the camera to same the config for.
        :param camera_config: Configuration to persist.
        """
        target_config_file = self._get_config_filename(camera_name)

        with open(target_config_file, 'w') as data_file:
            json.dump(camera_config, data_file, indent=True)
