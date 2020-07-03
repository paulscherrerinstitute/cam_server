import socket
import uuid
from logging import getLogger

from cam_server import config
from cam_server.instance_management.management import InstanceManager, InstanceWrapper
from cam_server.pipeline.configuration import PipelineConfig
from cam_server.pipeline.transceiver import get_pipeline_function
from cam_server.utils import update_pipeline_config

_logger = getLogger(__name__)


class PipelineInstanceManager(InstanceManager):
    def __init__(self, config_manager, background_manager, user_scripts_manager,
                 cam_server_client, hostname=None, port_range=None):
        super(PipelineInstanceManager, self).__init__(
            port_range=config.PIPELINE_STREAM_PORT_RANGE if (port_range is None) else port_range,
            auto_delete_stopped=True)
        self.config_manager = config_manager
        self.background_manager = background_manager
        self.user_scripts_manager = user_scripts_manager
        self.cam_server_client = cam_server_client
        self.hostname = hostname


    def get_pipeline_list(self):
        return self.config_manager.get_pipeline_list()

    def _create_and_start_pipeline(self, instance_id, pipeline_config, read_only_pipeline):
        if pipeline_config.get_pipeline_type() == config.PIPELINE_TYPE_STREAM:
            camera_name = None
        else:
            camera_name = pipeline_config.get_camera_name()
            if not self.cam_server_client.is_camera_online(camera_name):
                raise ValueError("Camera %s is not online. Cannot start pipeline." % camera_name)

        if pipeline_config.get_configuration().get("port"):
            stream_port =  int(pipeline_config.get_configuration().get("port"))
            _logger.info("Creating pipeline on fixed port '%s' for camera '%s'. instance_id=%s" %
                     (stream_port, camera_name, instance_id))
        else:
            stream_port = self.get_next_available_port(instance_id)
            _logger.info("Creating pipeline on port '%s' for camera '%s'. instance_id=%s" %
                     (stream_port, camera_name, instance_id))


        self.add_instance(instance_id, PipelineInstance(
            instance_id=instance_id,
            process_function=get_pipeline_function(pipeline_config.get_pipeline_type()),
            pipeline_config=pipeline_config,
            stream_port=stream_port,
            cam_client=self.cam_server_client,
            background_manager=self.background_manager,
            user_scripts_manager=self.user_scripts_manager,
            hostname=self.hostname,
            read_only_config=read_only_pipeline
        ))

        self.start_instance(instance_id)

    def create_pipeline(self, pipeline_name=None, configuration=None, instance_id=None):

        """
        Create the pipeline stream address. Either pass the pipeline name, or the configuration.
        :param pipeline_name: Name of the pipeline to load from config.
        :param configuration: Configuration to load the pipeline with.
        :param instance_id: Name to assign to the instace. It must be unique.
        :return: instance_id, Pipeline stream address.
        """

        # User specified or random uuid as the instance id.
        if not instance_id:
            instance_id = str(uuid.uuid4())

        if self.is_instance_present(instance_id):
            raise ValueError("Instance with id '%s' is already present and running. "
                             "Use another instance_id or stop the current instance "
                             "if you want to reuse the same instance_id." % instance_id)

        if (not pipeline_name) and (not configuration):
            raise ValueError("You must specify either the pipeline name or the configuration for the pipeline.")

        if pipeline_name:
            pipeline_config = self.config_manager.load_pipeline(pipeline_name)
            if configuration is not None:
                pipeline_config.update( configuration )
        else:
            pipeline_config = PipelineConfig(instance_id, configuration)

        self._create_and_start_pipeline(instance_id, pipeline_config, read_only_pipeline=False)

        pipeline_instance = self.get_instance(instance_id)

        return pipeline_instance.get_instance_id(), pipeline_instance.get_stream_address()

    def get_instance_stream(self, instance_id):
        if self.is_instance_present(instance_id):
            return self.get_instance(instance_id).get_stream_address()

        try:
            pipeline_config = self.config_manager.load_pipeline(instance_id)
        except ValueError:
            raise ValueError("Instance '%s' is not present on server and it is not a saved pipeline name." %
                             instance_id)

        self._create_and_start_pipeline(instance_id, pipeline_config, read_only_pipeline=True)

        return self.get_instance(instance_id).get_stream_address()

    def _find_instance_id_with_config(self, pipeline_config):
        # Search for read-only pipelines with the same config.
        for read_only_instance in (x for x in self.instances.values() if x.is_read_only_config()):
            if pipeline_config == read_only_instance.pipeline_config:
                return read_only_instance.get_instance_id()

        return None

    def get_instance_stream_from_config(self, configuration):
        pipeline_config = PipelineConfig(None, parameters=configuration)

        instance_id = self._find_instance_id_with_config(pipeline_config)
        if self.is_instance_present(instance_id):
            return instance_id, self.get_instance(instance_id).get_stream_address()
        else:
            # If the instance is not present, it was deleted because it was stopped.
            instance_id = None

        if instance_id is None:
            instance_id = str(uuid.uuid4())

        self._create_and_start_pipeline(instance_id, pipeline_config, read_only_pipeline=True)

        return instance_id, self.get_instance(instance_id).get_stream_address()

    def update_instance_config(self, instance_id, config_updates):
        pipeline_instance = self.get_instance(instance_id)

        current_config = pipeline_instance.get_configuration()

        image_background_enable = config_updates.get("image_background_enable")
        if not image_background_enable:
            image_background_enable = current_config.get("image_background_enable")

        # Check if the background can be loaded.
        if image_background_enable:
            image_background = config_updates.get("image_background")
            if image_background:
                self.background_manager.get_background(image_background)
        new_config = update_pipeline_config(current_config, config_updates)
        pipeline_instance.set_parameter(new_config)
        _logger.info("Instance config updated: %s" % (instance_id,))

    def get_instance_configuration(self, instance_id):
        return self.get_instance(instance_id).get_configuration()

    def get_instance_info(self, instance_id):
        return self.get_instance(instance_id).get_info()

    def save_pipeline_config(self, pipeline_name, config):
        return self.config_manager.save_pipeline_config(pipeline_name, config)
    
    def collect_background(self, camera_name, number_of_images):
        return self.background_manager.collect_background(self.cam_server_client, camera_name, number_of_images)

    def save_script(self, script_name, script):
        return self.user_scripts_manager.save_script(script_name, script)


class PipelineInstance(InstanceWrapper):
    def __init__(self, instance_id, process_function, pipeline_config, stream_port, cam_client,
                 background_manager, user_scripts_manager, hostname=None, read_only_config=False):

        super(PipelineInstance, self).__init__(instance_id, process_function, stream_port,
                                               cam_client, pipeline_config, stream_port,
                                               background_manager, user_scripts_manager )

        self.pipeline_config = pipeline_config

        if not hostname:
            hostname = socket.gethostname()

        self.stream_address = "tcp://%s:%d" % (hostname, stream_port)
        self.read_only_config = read_only_config


    def get_info(self):
        return {"stream_address": self.stream_address,
                "is_stream_active": self.is_running(),
                "camera_name": self.get_camera_name(),
                "config": self.get_configuration(),
                "instance_id": self.get_instance_id(),
                "read_only": self.read_only_config,
                "last_start_time": self.last_start_time,
                "statistics": self.get_statistics()
                }

    def get_configuration(self):
        return self.pipeline_config.get_configuration()

    def get_stream_address(self):
        return self.stream_address

    def get_camera_name(self):
        return self.pipeline_config.get_camera_name()

    def set_parameter(self, configuration):
        if self.read_only_config:
            raise ValueError("Cannot set config on a read only instance.")

        if self.is_running() and self.pipeline_config.get_camera_name() != configuration.get("camera_name"):
            raise ValueError("Cannot change the camera name on a running instance. Stop the instance first.")

        self.pipeline_config.set_configuration(configuration)

        # The set configuration sets the default parameters.
        super().set_parameter(self.pipeline_config.get_configuration())

    def get_parameters(self):
        return self.pipeline_config.get_configuration()

    def get_name(self):
        return self.pipeline_config.get_name()

    def is_read_only_config(self):
        return self.read_only_config
