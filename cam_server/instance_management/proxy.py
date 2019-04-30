import logging
import os
from concurrent.futures import ThreadPoolExecutor
from cam_server import config
from bottle import static_file


_logger = logging.getLogger(__name__)


class ProxyBase:
    def __init__(self, server_pool, configuration, config_manager):
        self.server_pool = server_pool
        self.configuration = configuration
        self.config_manager = config_manager
        self.default_server = server_pool[0]
        self.executor = ThreadPoolExecutor(len(self.server_pool))

    def register_rest_interface(self, app):
        api_root_address = config.API_PREFIX + config.PROXY_REST_INTERFACE_PREFIX

        @app.get(api_root_address + "/servers")
        def get_servers():
            """
            Return the list of servers and the load for each
            :return:
            """
            info = self.get_servers_info()
            status = {server: (info[server]['active_instances'] if info[server] else None) for server in info}
            servers = [server.get_address() for server in self.server_pool]
            instances, cpu, memory, tx, rx = [], [], [], [], []
            for server in self.server_pool:
                try:
                    instances.append(list(status[server.get_address()].keys()))
                except:
                    instances.append([])

            ret =  {"state": "ok",
                    "status": "List of servers.",
                    "servers": servers,
                    "load":  self.get_load(status),
                    "cpu":   [(info[server].get("cpu")    if info[server] else None) for server in info] if info else [],
                    "memory":[(info[server].get("memory") if info[server] else None) for server in info] if info else [],
                    "tx":    [(info[server].get("tx")     if info[server] else None) for server in info] if info else [],
                    "rx":    [(info[server].get("rx")     if info[server] else None) for server in info] if info else [],
                    "instances":  instances}
            return ret


        @app.get(api_root_address + "/status")
        def get_status():
            """
            Return the instances running in each server
            :return:
            """
            return {"state": "ok",
                    "status": "Status of available servers.",
                    "servers": self.get_status()}

        @app.get(api_root_address + "/info")
        def get_info():
            """
            Return the active instances obn all servers
            :return:
            """
            return {"state": "ok",
                    "status": "Running instances information.",
                    "info":  self.get_info()}

        @app.delete(api_root_address + "/<instance_name>")
        def stop_instance(instance_name):
            """
            Stop a specific camera.
            :param instance_name: Name of the camera.
            """
            self.stop_instance(instance_name)

            return {"state": "ok",
                    "status": "Instance '%s' stopped." % instance_name}

        @app.delete(api_root_address + "/server/<server_index>")
        def stop_all_server_instances(server_index):
            """
            Stop a specific camera.
            :param instance_name: Name of the camera.
            """
            server = self.server_pool[int(server_index)]
            self.stop_all_server_instances(server)
            return {"state": "ok",
                    "status": "All instances stopped in '%s'." % server.get_address()}


    def _get_root(self):
        return os.path.dirname(__file__)

    def register_management_page(self, app):
        @app.route('/')
        def home():
            return static_file("index.html", self._get_root())

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

    def get_servers_info(self):
            def task(server):
                try:
                    info = server.get_server_info()
                except:
                    info = None
                return (server, info)

            futures = []
            for server in self.server_pool:
                futures.append(self.executor.submit(task, server))

            ret = {}
            for future in futures:
                (server, info) = future.result()
                ret[server.get_address()] = info
            return ret

    def get_fixed_server(self, name, status=None):
        if status is None:
            status = self.get_status()
        for server in self.configuration.keys():
            try:
                if name in self.configuration[server]["instances"]:
                    return self.get_server_from_address(server)
            except:
                pass

    def get_server(self, instance_name=None, status=None):
        """
        If instance name is None returns the default server
        """
        if instance_name is None:
            return self.default_server
        if status is None:
            status = self.get_status()

        for server in self.server_pool:
            try:
                info = status[server.get_address()]
                if instance_name in info.keys():
                    return server
            except:
                pass
        return None

    def get_server_from_address(self, address):
        for server in self.server_pool:
            if address == server.get_address():
                return server
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

    def get_free_server(self, instance_name=None, status=None):
        load = self.get_load(status)
        m = min(load)
        if m >= 1000:
            raise Exception("No available server")
        return self.server_pool[load.index(m)]

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

    def get_instance_stream(self, instance_name):
        status = self.get_status()
        server = self.get_server(instance_name, status)
        if server is None:
            server = self.get_fixed_server(instance_name, status)
            if server is None:
                server = self.get_free_server(instance_name, status)
                _logger.info("Creating stream to %s at %s", instance_name, server.get_address())
            else:
                _logger.info("Creating fixed stream to %s at %s", instance_name, server.get_address())
        else:
            _logger.info("Connecting to stream %s at %s", instance_name, server.get_address())
        return server.get_instance_stream(instance_name)

    def stop_all_instances(self):
        _logger.info("Stopping all")
        for server in self.server_pool:
            try:
                server.stop_all_instances()
            except:
                pass

    def stop_all_server_instances(self, server):
        _logger.info("Stopping all instances at %s", server.get_address())
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