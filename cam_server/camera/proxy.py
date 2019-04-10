import logging
from concurrent.futures import ThreadPoolExecutor

_logger = logging.getLogger(__name__)


class Proxy:
    def __init__(self, config_manager, server_pool):
        self.config_manager = config_manager
        self.server_pool = server_pool
        self.default_server = server_pool[0]
        self.executor = ThreadPoolExecutor(len(self.server_pool))

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
                server.stop_all_cameras()
            except:
                pass

    def stop_instance(self, camera_name):
        server = self.get_server(camera_name)
        if server is not None:
            _logger.info("Stopping %s at %s", camera_name, server.get_address())
            server.stop_camera(camera_name)

    def get_camera_list(self):
        return self.default_server.get_cameras()

    def get_camera_stream(self, camera_name):
        status = self.get_status()
        server = self.get_server(camera_name, status)
        if server is None:
            server = self.get_free_server(status)
            _logger.info("Creating stream to %s at %s", camera_name, server.get_address())
        else:
            _logger.info("Connecting to stream %s at %s", camera_name, server.get_address())
        return server.get_camera_stream(camera_name)

    def set_camera_instance_config(self, camera_name, new_config):
        server = self.get_server(camera_name)
        if server is not None:
            #TODO: the config will be written twice
            server.set_camera_config(camera_name, new_config)

    #TODO: get_camera_image and get_camera_image_bytes conect diretly to the camera.
    #      What should do if there is a connected server to that camera?


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

    def get_server(self, camera_name, status=None):
        if status is None: status = self.get_status()
        for server in self.server_pool:
            try:
                info = status[server.get_address()]
                if camera_name in info.keys():
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




