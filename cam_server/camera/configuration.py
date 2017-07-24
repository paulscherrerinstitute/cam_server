from collections import OrderedDict

from cam_server.camera.receiver import CameraSimulation, Camera
from cam_server.instance_management.configuration import validate_camera_config


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
            camera_config = {"camera": {"prefix": "simulation"}}
        else:
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


class CameraConfig:

    def __init__(self, camera_name, parameters=None):
        self.camera_name = camera_name
        self.parameters = {}

        if parameters:
            self.parameters = parameters
        else:
            self.parameters = OrderedDict({
                "prefix": camera_name
            })

    @staticmethod
    def validate_config(config):
        # TODO: Implement validation.
        pass
