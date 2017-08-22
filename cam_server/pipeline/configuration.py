import glob
from collections import OrderedDict

import copy
import os
from datetime import datetime
from os.path import basename

import numpy


class PipelineConfigManager(object):
    def __init__(self, config_provider):
        self.config_provider = config_provider

    def get_pipeline_list(self):
        """
        Retrieve the list of available pipelines.
        :return: List of pipelines.
        """
        configured_pipelines = self.config_provider.get_available_configs()
        return configured_pipelines

    def get_pipeline_config(self, pipeline_name):
        """
        Return the pipeline configuration.
        :param pipeline_name: Name of the pipeline to retrieve the config for.
        :return: Pipeline config.
        """

        configuration = self.config_provider.get_config(pipeline_name)
        pipeline_config = PipelineConfig(pipeline_name, configuration)

        return pipeline_config.get_configuration()

    def load_pipeline(self, pipeline_name):
        """
        Load a cam_server with the given name.
        :param pipeline_name: Pipeline to load.
        :return: Pipeline instance.
        """
        configuration = self.get_pipeline_config(pipeline_name)
        return PipelineConfig(pipeline_name, configuration)

    def save_pipeline_config(self, pipeline_name, configuration):
        """
        Save the pipeline config changes.
        :param pipeline_name: Name of the cam_server to save the config to.
        :param configuration: Config to save.
        """

        pipeline_config = PipelineConfig(pipeline_name, configuration)
        self.config_provider.save_config(pipeline_name, pipeline_config.get_configuration())

    def delete_pipeline_config(self, pipeline_name):
        """
        Delete the pipeline config.
        :param pipeline_name: Name of the pipeline to delete.
        """

        if pipeline_name not in self.get_pipeline_list():
            raise ValueError("Pipeline '%s' does not exist." % pipeline_name)

        self.config_provider.delete_config(pipeline_name)


class BackgroundImageManager(object):
    def __init__(self, background_folder):

        if len(background_folder) > 1 and background_folder[-1] == '/':
            background_folder = background_folder[:-1]

        self.background_folder = background_folder

    def get_background(self, background_name):
        if not background_name:
            return None

        background_filename = os.path.join(self.background_folder, background_name + ".npy")

        if not os.path.exists(background_filename):
            raise ValueError("Requested background '%s' does not exist." % background_name)

        return numpy.load(background_filename)

    def save_background(self, background_name, image, append_timestamp=True):
        if append_timestamp:
            background_name += datetime.now().strftime("_%Y%m%d_%H%M%S_%f")

        background_filename = os.path.join(self.background_folder, background_name + ".npy")
        numpy.save(background_filename, image)

        return background_name

    def get_latest_background_id(self, background_prefix):
        matching_backgrounds = glob.glob(self.background_folder + '/%s*.npy' % background_prefix)

        if not matching_backgrounds:
            raise ValueError("No background matches for the specified prefix '%s'." % background_prefix)

        latest_background_filename = sorted(matching_backgrounds)[-1]
        latest_background_id = os.path.splitext(basename(latest_background_filename))[0]

        return latest_background_id


class PipelineConfig:

    DEFAULT_CONFIGURATION = {
        "image_background_enable": False,
        "image_background": None,
        "image_threshold": None,
        "image_region_of_interest": None,
        "image_good_region": None,
        "image_slices": None
    }

    DEFAULT_IMAGE_GOOD_REGION = {
        "threshold": 0.3,
        "gfscale": 1.8
    }

    DEFAULT_IMAGE_SLICES = {
        "number_of_slices": 1,
        "scale": 2
    }

    def __init__(self, pipeline_name, parameters=None):

        self.pipeline_name = pipeline_name

        if parameters is not None:
            self.parameters = parameters
        else:
            # Default pipeline parameters.
            self.parameters = OrderedDict(
                {
                    "camera_name": "simulation"
                })

        # Expand the config with the default values.
        self.parameters = PipelineConfig.expand_config(self.parameters)

        # Check if the config is valid.
        PipelineConfig.validate_pipeline_config(self.parameters)

    def get_configuration(self):
        # Validate before passing on, since anyone can change the dictionary content.
        PipelineConfig.validate_pipeline_config(self.parameters)
        # We do not want to pass by reference - someone might change the dictionary.
        return copy.deepcopy(self.parameters)

    def set_configuration(self, configuration):
        new_parameters = PipelineConfig.expand_config(configuration)
        PipelineConfig.validate_pipeline_config(new_parameters)
        self.parameters = new_parameters

    def get_background_id(self):
        return self.parameters.get("image_background")

    @staticmethod
    def validate_pipeline_config(configuration):
        """
        Verify if the pipeline config has all the mandatory attributes.
        :param configuration: Config to validate.
        :return:
        """

        if not configuration:
            raise ValueError("Config object cannot be empty.\nConfig: %s" % configuration)

        if "camera_name" not in configuration:
            raise ValueError("Camera name not specified in configuration.")

        def verify_attributes(section_name, section, mandatory_attributes):
            missing_attributes = [attr for attr in mandatory_attributes if attr not in section]

            if missing_attributes:
                raise ValueError("The following mandatory attributes were not found in the %s: %s" %
                                 (section_name, missing_attributes))

        # Verify root attributes.
        verify_attributes("configuration", configuration, PipelineConfig.DEFAULT_CONFIGURATION.keys())

        image_good_region = configuration["image_good_region"]
        if image_good_region:
            verify_attributes("image_good_region", image_good_region, PipelineConfig.DEFAULT_IMAGE_GOOD_REGION)

        image_slices = configuration["image_slices"]
        if image_slices:
            verify_attributes("image_slices", image_slices, PipelineConfig.DEFAULT_IMAGE_SLICES)

    @staticmethod
    def expand_config(configuration):

        def expand_section(provided_value, default_value):
            expanded_section = copy.deepcopy(default_value)
            # Prevent None values to overwrite defaults.
            expanded_section.update((k, v) for k, v in provided_value.items() if v is not None)
            return expanded_section

        expanded_config = expand_section(configuration, PipelineConfig.DEFAULT_CONFIGURATION)

        if expanded_config["image_good_region"] is not None:
            expanded_config["image_good_region"] = expand_section(expanded_config["image_good_region"],
                                                                  PipelineConfig.DEFAULT_IMAGE_GOOD_REGION)

        if expanded_config["image_slices"] is not None:
            expanded_config["image_slices"] = expand_section(expanded_config["image_slices"],
                                                             PipelineConfig.DEFAULT_IMAGE_SLICES)

        return expanded_config

    def get_name(self):
        return self.pipeline_name

    def get_camera_name(self):
        return self.parameters["camera_name"]
