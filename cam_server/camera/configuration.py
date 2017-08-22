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

        self.config_provider.save_config(camera_name, camera_config.get_configuration())

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

    DEFAULT_CONFIGURATION = {
        "camera_calibration": None,
        "mirror_x": False,
        "mirror_y": False,
        "rotate": 0
    }

    DEFAULT_CAMERA_CALIBRATION = {
        "reference_marker": [0, 0, 100, 100],
        "reference_marker_width": 100.0,
        "reference_marker_height": 100.0,
        "angle_horizontal": 0.0,
        "angle_vertical": 0.0
    }

    def __init__(self, camera_name, parameters=None):
        self.camera_name = camera_name
        self.parameters = {}

        if parameters is not None:
            self.parameters = parameters
        else:
            self.parameters = OrderedDict({
                "prefix": camera_name
            })

        # Expand the config with the default values.
        self.parameters = CameraConfig.expand_config(self.parameters)

        CameraConfig.validate_camera_config(self.parameters)

    def get_configuration(self):
        # Validate before passing on, since anyone can change the dictionary content.
        self.validate_camera_config(self.parameters)
        # We do not want to pass by reference - someone might change the dictionary.
        return copy.deepcopy(self.parameters)

    def set_configuration(self, configuration):
        new_parameters = CameraConfig.expand_config(configuration)
        CameraConfig.validate_camera_config(new_parameters)
        self.parameters = new_parameters

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

        if "prefix" not in configuration:
            raise ValueError("Prefix not specified in configuration.")

        def verify_attributes(section_name, section, mandatory_attributes):
            missing_attributes = [attr for attr in mandatory_attributes if attr not in section]

            if missing_attributes:
                raise ValueError("The following mandatory attributes were not found in the %s: %s" %
                                 (section_name, missing_attributes))

        # Verify root attributes.
        verify_attributes("configuration", configuration, CameraConfig.DEFAULT_CONFIGURATION.keys())

        camera_calibration = configuration["camera_calibration"]
        if camera_calibration:
            verify_attributes("camera_calibration", camera_calibration, CameraConfig.DEFAULT_CAMERA_CALIBRATION)

    @staticmethod
    def expand_config(configuration):

        def expand_section(provided_value, default_value):
            expanded_section = copy.deepcopy(default_value)
            # Prevent None values to overwrite defaults.
            expanded_section.update((k, v) for k, v in provided_value.items() if v is not None)
            return expanded_section

        # No expansion of default parameters.
        expanded_config = expand_section(configuration, CameraConfig.DEFAULT_CONFIGURATION)

        if expanded_config["camera_calibration"] is not None:
            expanded_config["camera_calibration"] = expand_section(expanded_config["camera_calibration"],
                                                                   CameraConfig.DEFAULT_CAMERA_CALIBRATION)

        return expanded_config
