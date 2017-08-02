from collections import OrderedDict

import copy


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

        PipelineConfig.validate_pipeline_config(configuration)

        return configuration

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

        PipelineConfig.validate_pipeline_config(configuration)
        self.config_provider.save_config(pipeline_name, configuration)

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
        self.background_folder = background_folder

    def get_background(self, background_name):
        pass

    def save_background(self, background_name, image):
        pass


class PipelineConfig:
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

        # Check if the config is valid.
        self.validate_pipeline_config(self.parameters)

    def get_parameters(self):
        # Validate before passing on, since anyone can change the dictionary content.
        self.validate_pipeline_config(self.parameters)
        # We do not want to pass by reference - someone might change the dictionary.
        return copy.deepcopy(self.parameters)

    @staticmethod
    def validate_pipeline_config(configuration):
        """
        Verify if the pipeline config has all the mandatory attributes.
        :param configuration: Config to validate.
        :return:
        """

        if not configuration:
            raise ValueError("Config object cannot be empty.\nConfig: %s" % configuration)

        mandatory_attributes = ["camera_name"]
        missing_attributes = [attr for attr in mandatory_attributes if attr not in configuration]

        if missing_attributes:
            raise ValueError("The following mandatory attributes were not found in the configuration: %s" %
                             missing_attributes)

    def get_name(self):
        return self.pipeline_name

    def get_camera_name(self):
        return self.parameters["camera_name"]
