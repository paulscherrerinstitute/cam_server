import logging
from concurrent.futures import ThreadPoolExecutor

_logger = logging.getLogger(__name__)


class Proxy:
    def __init__(self, config_manager, background_manager, cam_server_client, server_pool):
        self.config_manager = config_manager
        self.server_pool = server_pool
        self.default_server = server_pool[0]
        self.executor = ThreadPoolExecutor(len(self.server_pool))
        self.background_manager = background_manager
        self.cam_server_client = cam_server_client

    def get_info(self):
        ret = {'active_instances':{}}
        status = self.get_status()
        for server in status.keys():
            info = status[server]
            if info is not None:
                for k in info.keys():
                    info[k]["host"] = server
                ret['active_instances'].update(info)
        return ret

    def stop_all_instances(self):
        _logger.info("Stopping all")
        for server in self.server_pool:
            try:
                server.stop_all_instances()
            except:
                pass

    def stop_instance(self, instance_name):
        status = self.get_status()
        server = self.get_server(instance_name, status)
        if server is not None:
            _logger.info("Stopping %s at %s", instance_name, server.get_address())
            server.stop_instance(instance_name)

    def get_pipeline_list(self):
        return self.default_server.get_pipelines()

    def create_pipeline(self, pipeline_name=None, configuration=None, instance_id=None):
        if pipeline_name is not None:
            instance_id, stream_address = self.default_server.create_instance_from_name(pipeline_name, instance_id)
        elif configuration is not None:
            instance_id, stream_address = self.default_server.create_instance_from_config(configuration, instance_id)
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

    def get_instance_stream(self, instance_name):
        status = self.get_status()
        server = self.get_server(instance_name, status)
        if server is None:
            server = self.get_free_server(status)
            _logger.info("Creating stream to %s at %s", instance_name, server.get_address())
        else:
            _logger.info("Connecting to stream %s at %s", instance_name, server.get_address())
        return server.get_instance_stream(instance_name)


    def get_instance_stream_from_config(self, configuration):
        #TODO
        status = self.get_status()
        #server = self.get_server_for_camera(camera_name, status)
        server = self.default_server
        if server is None:
            server = self.get_free_server(status)
        return server.get_instance_stream_from_config(configuration)


    def update_instance_config(self, instance_name, config_updates):
        server = self.get_server(instance_name)
        if server is not None:
            #TODO: the config will be written twice
            server.set_instance_config(instance_name, config_updates)




    def get_status(self):
        def task(server):
            try:
                instances = server.get_server_info()['active_instances']
            except:
                instances = None
            return (server,instances)
        futures = []
        for server in self.server_pool:
            futures.append(self.executor.submit(task, server))

        ret = {}
        for future in futures:
            (server,instances) = future.result()
            ret[server.get_address()] = instances
        return ret

    def get_server(self, instance_name, status=None):
        if status is None: status = self.get_status()
        for server in self.server_pool:
            try:
                info = status[server.get_address()]
                if instance_name in info.keys():
                    return server
            except:
                pass
        return None

    def get_load(self, status=None):
        if status is None: status = self.get_status()
        load = []
        for server in self.server_pool:
            try:
                load.append(len(status[server.get_address()]))
            except:
                load.append(1000)
        return load

    def get_free_server(self, status=None):
        load = self.get_load(status)
        m = min(load)
        if m >= 1000:
            raise Exception("No available server")
        return self.server_pool[load.index(m)]