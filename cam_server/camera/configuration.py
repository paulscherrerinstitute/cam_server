import copy
from cam_server.camera.source.utils import get_source_class, source_type_to_source_class_mapping


class CameraConfigManager(object):
    def __init__(self, config_provider):
        self.config_provider = config_provider

    def get_camera_list(self):
        """
        Retrieve the list of available cameras.
        :return: List of cameras.
        """
        configured_cameras = self.config_provider.get_available_configs()

        return configured_cameras

    def get_camera_config(self, camera_name):
        """
        Return the camera configuration.
        :param camera_name: Name of the cam_server to retrieve the config for.
        :return: Camera config dictionary.
        """
        configuration = self.config_provider.get_config(camera_name)
        return CameraConfig(camera_name, parameters=configuration)

    def delete_camera_config(self, camera_name):
        """
        Delete an existing config.
        :param camera_name: Name of the camera to delete.
        """

        if camera_name not in self.get_camera_list():
            raise ValueError("Config '%s' does not exist." % camera_name)

        self.config_provider.delete_config(camera_name)

    def load_camera(self, camera_name):
        """
        Load a camera with the given name.
        :param camera_name: Camera to load.
        :return: Camera instance.
        """
        camera_config = self.get_camera_config(camera_name)
        camera_class = get_source_class(camera_config.get_source_type())

        return camera_class(camera_config)

    def save_camera_config(self, camera_name, config):
        """
        Save the camera config changes.
        :param camera_name: Name of the cam_server to save the config to.
        :param config: Config to save.
        """
        CameraConfig.validate_camera_config(config)

        self.config_provider.save_config(camera_name, config)

    def get_camera_geometry(self, camera_name):
        """
        Returns the camera geometry info.
        :param camera_name: Name of the camera.
        :return: (width, height)
        """
        camera = self.load_camera(camera_name)
        width, height = camera.get_geometry()
        return width, height


class CameraConfig:

    MANDATORY_CONFIGURATION = ["source_type"]

    DEFAULT_CONFIGURATION = {
        "camera_calibration": None,
        "mirror_x": False,
        "mirror_y": False,
        "rotate": 0,
        "roi": None,
        "image_background":None,
        "source_type": "epics"
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
            self.parameters = {"source": "simulation",
                               "source_type": "simulation"}

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

    def get_source_type(self):
        return self.parameters["source_type"]

    def get_source(self):
        return self.parameters["source"]

    @staticmethod
    def validate_camera_config(configuration):
        """
        Verify if the camera config has the mandatory attributes.
        :param configuration: Configuration to verify.
        :return:
        """
        if not configuration:
            raise ValueError("Config object cannot be empty. Config: %s" % configuration)

        if "source" not in configuration:
            raise ValueError("'source' not specified in configuration.")

        def verify_attributes(section_name, section, mandatory_attributes):
            missing_attributes = [attr for attr in mandatory_attributes if attr not in section]

            if missing_attributes:
                raise ValueError("The following mandatory attributes were not found in the %s: %s" %
                                 (section_name, missing_attributes))

        # Verify root attributes.
        verify_attributes("configuration", configuration, CameraConfig.MANDATORY_CONFIGURATION)

        camera_calibration = configuration["camera_calibration"]
        if camera_calibration:
            verify_attributes("camera_calibration", camera_calibration, CameraConfig.DEFAULT_CAMERA_CALIBRATION)

        available_source_types = source_type_to_source_class_mapping.keys()

        if configuration["source_type"] not in available_source_types:
            raise ValueError("Invalid source_type '%s'. Available: %s." % (configuration["source_type"],
                                                                           list(available_source_types)))

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
