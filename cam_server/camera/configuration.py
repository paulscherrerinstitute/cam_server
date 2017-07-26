from collections import OrderedDict

import copy

from cam_server.camera.receiver import CameraSimulation, Camera


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
        if camera_name.lower() == "simulation":
            configuration = None
        else:
            configuration = self.config_provider.get_config(camera_name)

        return CameraConfig(camera_name, parameters=configuration)

    def delete_camera_config(self, camera_name):
        """
        Delete an existing config.
        :param camera_name: Name of the camera to delete.
        """

        # Simulation cam_server is not defined in the config.
        if camera_name.lower() == "simulation":
            raise ValueError("Cannot delete simulation camera.")

        if camera_name not in self.get_camera_list():
            raise ValueError("Config '%s' does not exist." % camera_name)

        self.config_provider.delete_config(camera_name)

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

        return Camera(camera_config)

    def save_camera_config(self, camera_name, config_updates):
        """
        Save the camera config changes.
        :param camera_name: Name of the cam_server to save the config to.
        :param config_updates: Config to save.
        """

        if camera_name.lower() == 'simulation':
            raise ValueError("Cannot save config for simulation camera.")

        # Get either the existing config, or generate a template one.
        try:
            camera_config = self.get_camera_config(camera_name)
        except ValueError:
            # Config does not exist, create an empty template.
            camera_config = CameraConfig(camera_name)

        camera_config.parameters.update(config_updates)

        self.config_provider.save_config(camera_name, camera_config.to_dict())

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
                "prefix": camera_name,
                "mirror_x": False,
                "mirror_y": False,
                "rotate": 0
            })

        self.validate_camera_config(self.parameters)

    def to_dict(self):
        # Validate before passing on, since anyone can change the dictionary content.
        self.validate_camera_config(self.parameters)
        # We do not want to pass by reference - someone might change the dictionary.
        return copy.deepcopy(self.parameters)

    def get_name(self):
        return self.camera_name

    @staticmethod
    def validate_camera_config(configuration):
        """
        Verify if the camera config has the mandatory attributes.
        :param configuration: Configuration to verify.
        :return:
        """
        if not configuration:
            raise ValueError("Config object cannot be empty.\nConfig: %s" % configuration)

        mandatory_attributes = ["prefix", "mirror_x", "mirror_y", "rotate"]
        missing_attributes = [attr for attr in mandatory_attributes if attr not in configuration]

        if missing_attributes:
            raise ValueError("The following mandatory attributes were not found in the configuration: %s" %
                             missing_attributes)
