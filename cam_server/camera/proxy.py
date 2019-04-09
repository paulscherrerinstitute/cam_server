import logging


_logger = logging.getLogger(__name__)


class Proxy:
    def __init__(self, config_manager, sever_pool):
        self.config_manager = config_manager
        self.sever_pool = sever_pool
        self.default_server = sever_pool[0]

    def get_server(self, camera_name):
        for server in self.sever_pool:
            try:
                info = server.get_server_info()['active_instances']
                if camera_name in info.keys():
                    return server
            except:
                pass
        return None

    def get_info(self):
        ret = {'active_instances':{}}
        for server in self.sever_pool:
            try:
                info = server.get_server_info()['active_instances']
                # Injecting host name
                for k in info.keys():
                    info[k]["host"] = server.get_address()
                ret['active_instances'].update(info)
            except:
                pass
        return ret

    def stop_all_instances(self):
        for server in self.sever_pool:
            try:
                server.stop_all_cameras()
            except:
                pass

    def stop_instance(self, instance_name):
        server = self.get_server(instance_name)
        if server is not None:
            server.stop_camera(instance_name)


    def get_camera_list(self):
        return self.default_server.get_cameras()

    def get_camera_stream(self, camera_name):
        return self.default_server.get_camera_stream(camera_name)

    def set_camera_instance_config(self, camera_name, new_config):
        server = self.get_server(camera_name)
        if server is not None:
            #TODO: the config will be written twice
            server.set_camera_config(camera_name, new_config)

    #TODO: get_camera_image and get_camera_image_bytes conect diretly to the camera.
    #      What should do if there is a connected server to that camera?