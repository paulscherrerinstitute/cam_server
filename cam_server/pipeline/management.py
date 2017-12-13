import socket
import uuid
from logging import getLogger

from cam_server import config
from cam_server.instance_management.management import InstanceManager, InstanceWrapper
from cam_server.pipeline.configuration import PipelineConfig
from cam_server.pipeline.transceiver import get_pipeline_function
from cam_server.utils import update_pipeline_config, get_port_generator

_logger = getLogger(__name__)


class PipelineInstanceManager(InstanceManager):
    def __init__(self, config_manager, background_manager, cam_server_client, hostname=None, port_range=None):
        super(PipelineInstanceManager, self).__init__()

        self.config_manager = config_manager
        self.background_manager = background_manager
        self.cam_server_client = cam_server_client
        self.hostname = hostname

        if port_range is None:
            port_range = config.PIPELINE_STREAM_PORT_RANGE

        self._port_generator = get_port_generator(port_range)
        self._used_ports = {}

    def _get_next_available_port(self, instance_id):
        # Clean up any stopped instances.
        instance_ids = list(self.instances.keys())
        for instance_id in instance_ids:
            self._delete_stopped_instance(instance_id)

        # Loop over all ports.
        for _ in range(*config.PIPELINE_STREAM_PORT_RANGE):
            candidate_port = next(self._port_generator)

            if candidate_port not in self._used_ports:
                self._used_ports[candidate_port] = instance_id
                return candidate_port

        raise Exception("All ports are used. Stop some instances before opening a new stream.")

    def _delete_stopped_instance(self, instance_id):
        # If instance is present but not running, delete it.
        if self.is_instance_present(instance_id) and not self.get_instance(instance_id).is_running():
            stream_port = self.get_instance(instance_id).get_stream_port()

            self.delete_instance(instance_id)

            del self._used_ports[stream_port]

    def get_pipeline_list(self):
        return self.config_manager.get_pipeline_list()

    def _create_and_start_pipeline(self, instance_id, pipeline_config, read_only_pipeline):
        stream_port = self._get_next_available_port(instance_id)

        camera_name = pipeline_config.get_camera_name()

        if not self.cam_server_client.is_camera_online(camera_name):
            raise ValueError("Camera %s is not online. Cannot start pipeline." % camera_name)

        _logger.info("Creating pipeline on port '%s' for camera '%s'. instance_id=%s",
                     stream_port, camera_name, instance_id)

        self.add_instance(instance_id, PipelineInstance(
            instance_id=instance_id,
            process_function=get_pipeline_function(pipeline_config.get_pipeline_type()),
            pipeline_config=pipeline_config,
            output_stream_port=stream_port,
            cam_client=self.cam_server_client,
            background_manager=self.background_manager,
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

        self._delete_stopped_instance(instance_id)

        if self.is_instance_present(instance_id):
            raise ValueError("Instance with id '%s' is already present and running. "
                             "Use another instance_id or stop the current instance "
                             "if you want to reuse the same instance_id." % instance_id)

        # You cannot specify both or none.
        if bool(pipeline_name) == bool(configuration):
            raise ValueError("You must specify either the pipeline name or the configuration for the pipeline.")

        if configuration:
            pipeline_config = PipelineConfig(instance_id, configuration)
        else:
            pipeline_config = self.config_manager.load_pipeline(pipeline_name)

        self._create_and_start_pipeline(instance_id, pipeline_config, read_only_pipeline=False)

        pipeline_instance = self.get_instance(instance_id)

        return pipeline_instance.get_instance_id(), pipeline_instance.get_stream_address()

    def get_instance_stream(self, instance_id):
        self._delete_stopped_instance(instance_id)

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
        self._delete_stopped_instance(instance_id)

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
        self._delete_stopped_instance(instance_id)

        pipeline_instance = self.get_instance(instance_id)

        current_config = pipeline_instance.get_configuration()

        # Check if the background can be loaded.
        image_background = config_updates.get("image_background")
        if image_background:
            self.background_manager.get_background(image_background)

        new_config = update_pipeline_config(current_config, config_updates)
        pipeline_instance.set_parameter(new_config)

    def stop_instance(self, instance_name):
        super().stop_instance(instance_name)
        self._delete_stopped_instance(instance_name)

    def stop_all_instances(self):
        _logger.info("Stopping all instances.")

        instance_ids = list(self.instances.keys())

        for instance_id in instance_ids:
            self.stop_instance(instance_id)


class PipelineInstance(InstanceWrapper):
    def __init__(self, instance_id, process_function, pipeline_config, output_stream_port, cam_client,
                 background_manager, hostname=None, read_only_config=False):

        super(PipelineInstance, self).__init__(instance_id, process_function,
                                               cam_client, pipeline_config, output_stream_port,
                                               background_manager)

        self.pipeline_config = pipeline_config

        if not hostname:
            hostname = socket.gethostname()

        self.stream_address = "tcp://%s:%d" % (hostname, output_stream_port)
        self.read_only_config = read_only_config
        self.stream_port = output_stream_port

    def get_info(self):
        return {"stream_address": self.stream_address,
                "is_stream_active": self.is_running(),
                "camera_name": self.pipeline_config.get_camera_name(),
                "config": self.pipeline_config.get_configuration(),
                "instance_id": self.get_instance_id(),
                "read_only": self.read_only_config,
                "last_start_time": self.last_start_time,
                "statistics": self.get_statistics()}

    def get_configuration(self):
        return self.pipeline_config.get_configuration()

    def get_stream_address(self):
        return self.stream_address

    def set_parameter(self, configuration):
        if self.read_only_config:
            raise ValueError("Cannot set config on a read only instance.")

        if self.is_running() and self.pipeline_config.get_camera_name() != configuration["camera_name"]:
            raise ValueError("Cannot change the camera name on a running instance. Stop the instance first.")

        self.pipeline_config.set_configuration(configuration)

        # The set configuration sets the default parameters.
        super().set_parameter(self.pipeline_config.get_configuration())

    def get_parameters(self):
        return self.pipeline_config.get_configuration()

    def get_name(self):
        return self.pipeline_config.get_name()

    def get_stream_port(self):
        return self.stream_port

    def is_read_only_config(self):
        return self.read_only_config
