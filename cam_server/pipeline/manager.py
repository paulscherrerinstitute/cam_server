import logging
import socket
from threading import Timer
from cam_server.instance_management.proxy import ProxyBase
from cam_server.pipeline.configuration import PipelineConfig
from cam_server import config
from cam_server import PipelineClient
from cam_server.utils import get_host_port_from_stream_address, cleanup

_logger = logging.getLogger(__name__)


class Manager(ProxyBase):
    def __init__(self, config_manager, background_manager, user_scripts_manager,
                 cam_server_client, config_str, bg_days_to_live=-1, client_timeout=None, update_timeout=None):
        ProxyBase.__init__(self, config_manager, config_str, PipelineClient, client_timeout, update_timeout)
        self.background_manager = background_manager
        self.cam_server_client = cam_server_client
        self.user_scripts_manager = user_scripts_manager
        self.update_timeout = update_timeout
        # background cleanup every day and upon start
        if bg_days_to_live>=0:
            def background_cleanup():
                self.cleanup_background_folder()
                self.timer = Timer(24*3600, background_cleanup)
                self.timer.daemon = True
                self.timer.start()
            background_cleanup()

    def get_current_servers_for_camera(self, camera, status=None):
        if not status:
            status = self.get_status()
        ret = []
        for server in status:
            if status[server]:
                for instance in status[server]:
                    if camera == status[server][instance]['camera_name']:
                        ret.append(self.get_server_from_address(server))
                        break
        return ret

    def get_running_camserver(self, camera, status):
        servers = []
        load = self.get_load(status)
        loads = []
        try:
            instances = self.cam_server_client.get_server_info(
                timeout = self.update_timeout if self.update_timeout else config.DEFAULT_SERVER_INFO_TIMEOUT)["active_instances"]
            host, port = get_host_port_from_stream_address(instances[camera]["host"])
            if not host.count(".") == 3: #If not IP, get only prefix
                host= host.split(".")[0].lower()
                local_names = ["127.0.0.1" "localhost"]
                local_host_name = socket.gethostname()
                if local_host_name:
                    local_names.append(local_host_name.split(".")[0].lower())

            for i in range(len(self.server_pool)):
                if load[i] < 1000:
                    try:
                        server_host, server_port = get_host_port_from_stream_address(self.server_pool[i].get_address())
                        if not host.count(".") == 3:  # If not IP, get only prefix
                            server_host = server_host.split(".")[0].lower() if server_host else None
                        if host==server_host:
                            servers.append(self.server_pool[i])
                            loads.append(load[i])
                        if (host in local_names) and (server_host in local_names):
                            servers.append(self.server_pool[i])
                            loads.append(load[i])
                    except:
                        pass
        except:
            pass
        if len(servers) > 0:
            m = min(loads)
            return servers[loads.index(m)]

        return None

    def get_server_for_camera(self, camera_name, status=None):
        if not status:
            status = self.get_status()
        servers = self.get_current_servers_for_camera(camera_name, status)
        if len(servers) > 0:
            _logger.info("Located running server for camera %s in the active pipelines" % camera_name)
            return servers[0]
        else:
            server = self.get_fixed_server_for_camera(camera_name, status)
            if server is not None:
                return server
            server = self.get_running_camserver(camera_name, status)
            if server is not None:
                _logger.info("Located running server for camera %s in the cam_server status" % camera_name)
                return server
            return self.get_free_server(None, status)

    def get_server_for_pipeline(self, pipeline_name, configuration, status=None):
        if not status:
            status = self.get_status()
        if pipeline_name is not None:
            server = self.get_fixed_server(pipeline_name, status)
            if server is not None:
                return server
        camera_name = configuration["camera_name"]
        return self.get_server_for_camera(camera_name, status)

    def get_pipeline_list(self):
        return self.config_manager.get_pipeline_list()

    def get_pipeline_last_backgrounds(self):
        ret = {}
        for pipeline in self.get_pipeline_list():
            try:
                ret[pipeline] = self.background_manager.get_latest_background_id(pipeline)
            except:
                ret[pipeline] = None
        return ret

    def get_last_background_filenames(self):
        return [(x + ".npy") for x in self.get_pipeline_last_backgrounds().values() if x is not None]

    def cleanup_background_folder(self, age_in_days = None, simulated=False):
        if age_in_days is None:
            age_in_days = self.bg_days_to_live
        if age_in_days>=0:
            last_backgrounds = self.get_last_background_filenames()
            path = self.background_manager.background_folder
            cleanup(age_in_days, path, False, False, last_backgrounds, simulated=simulated)


    def create_pipeline(self, pipeline_name=None, configuration=None, instance_id=None):
        """
        If both pipeline_name and configuration are set, pipeline is create from name and
        configuration field added as additional config parameters
        """
        status = self.get_status()

        if (not pipeline_name) and (not configuration):
            raise ValueError("You must specify either the pipeline name or the configuration for the pipeline.")

        if pipeline_name is not None:
            cfg = self.config_manager.get_pipeline_config(pipeline_name)
        elif configuration is not None:
            cfg = PipelineConfig.expand_config(configuration)
            PipelineConfig.validate_pipeline_config(cfg)

        server = None
        if instance_id is not None:
            server = self.get_server(instance_id, status)
        if server is None:
            server = self.get_server_for_pipeline(pipeline_name, cfg, status)

        self._check_background(server, cfg)
        if pipeline_name is not None:
            server.save_pipeline_config(pipeline_name, cfg)
            _logger.info("Creating stream from name %s at %s" % (pipeline_name, server.get_address()))
            instance_id, stream_address = server.create_instance_from_name(pipeline_name, instance_id, configuration)
        elif cfg is not None:
            _logger.info("Creating stream from config to camera %s at %s" % (cfg["camera_name"], server.get_address()))
            instance_id, stream_address = server.create_instance_from_config(cfg, instance_id)
        else:
            raise Exception("Invalid parameters")

        return instance_id, stream_address

    def get_instance_configuration(self, instance_name):
        server = self.get_server(instance_name)
        if server is not None:
            return server.get_instance_config(instance_name)
        raise ValueError("Instance '%s' does not exist." % instance_name)

    def get_instance_info(self, instance_name):
        server = self.get_server(instance_name)
        if server is not None:
            return server.get_instance_info(instance_name)
        raise ValueError("Instance '%s' does not exist." % instance_name)

    def get_instance_stream_from_config(self, configuration):
        status = self.get_status()
        server = self.get_server_for_pipeline(None, configuration, status)
        _logger.info("Getting stream from config to camera %s at %s" %
                     (configuration["camera_name"], server.get_address()))
        return server.get_instance_stream_from_config(configuration)

    def update_instance_config(self, instance_name, config_updates):
        server = self.get_server(instance_name)
        if server is not None:
            self._check_background(server, config_updates)
            server.set_instance_config(instance_name, config_updates)

    def collect_background(self, camera_name, number_of_images):
        background_id = self.background_manager.collect_background(self.cam_server_client, camera_name, number_of_images)
        for server in self.get_current_servers_for_camera(camera_name):
            image_array = self.background_manager.get_background(background_id)
            server.set_background_image_bytes(background_id, image_array)
        return background_id

    def save_script(self, script_name, script):
            if script_name  and script:
                self.user_scripts_manager.save_script(script_name, script)
                for server in self.server_pool:
                    try:
                        server.set_user_script( script_name, script)
                    except:
                        _logger.error("Error setting user script %s on %s" % (script_name, server.get_address()))

    def _check_background(self, server, config):
        if config.get("image_background_enable"):
            image_background = config.get("image_background")
            if image_background:
                # Check if the background can be loaded
                image_array = self.background_manager.get_background(image_background)
                server.set_background_image_bytes(image_background, image_array)

    def save_pipeline_config(self, pipeline_name, config):
        self.config_manager.save_pipeline_config(pipeline_name, config)
        for server in self.server_pool:
            try:
                server.save_pipeline_config(pipeline_name, config)
            except:
                pass

    def get_fixed_server_for_camera(self, name, status=None):
        if status is None:
            status = self.get_status()
        for server in self.configuration.keys():
            try:
                if name in self.configuration[server]["cameras"]:
                    return self.get_server_from_address(server)
            except:
                pass

    def start_permanent_instance(self, pipeline, name):
        _logger.info("Starting permanent instance of %s: %s" % (pipeline,name))

        self.create_pipeline(pipeline, {"no_client_timeout": 0}, name)

