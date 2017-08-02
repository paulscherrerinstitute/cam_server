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
        # TODO: Check for background, and add.
        # background_file = camera_configuration_directory + '/' + camera_name + '_background.npy'
        # if os.path.isfile(background_file):
        #     parameter.background_image = numpy.load(background_file)

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


class PipelineConfig:
    def __init__(self, pipeline_name, parameters=None):

        self.pipeline_name = pipeline_name

        if parameters is not None:
            self.parameters = parameters
        else:
            # Default pipeline parameters.
            self.parameters = OrderedDict(
                {"background_subtraction": False,

                 "apply_threshold": False,
                 "threshold": None,  # default 1.0

                 "apply_region_of_interest": False,
                 "region_of_interest": None,  # (offset_x, size_x, offset_y, size_y)

                 "apply_good_region": False,
                 "good_region_threshold": 0.3,  # default 0.3
                 "good_region_gfscale": 1.8,  # default 1.8

                 "apply_slices": False,
                 "slices_number": 1,
                 "slices_scale": 2  # width_all_slices = scale * standard_deviation
                 })

        self.background_image = None

        # Check if the config is valid.
        self.validate_pipeline_config(self.parameters)

    def to_dict(self):
        # Validate before passing on, since anyone can change the dictionary content.
        self.validate_pipeline_config(self.parameters)
        # We do not want to pass by reference - someone might change the dictionary.
        return copy.deepcopy(self.parameters)

    @staticmethod
    def validate_pipeline_config(config):
        # TODO: implement validation.
        pass

    def get_name(self):
        return self.pipeline_name
