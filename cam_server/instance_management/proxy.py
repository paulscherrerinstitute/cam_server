import json
import logging
import os
import socket
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Timer

from bottle import static_file, request, response

from cam_server import config, __VERSION__
from cam_server.utils import get_host_port_from_stream_address

_logger = logging.getLogger(__name__)


class ProxyBase:
    def __init__(self, config_manager, config_str, client_class, server_timeout = None, update_timeout = None):
        self.config_manager = config_manager
        self.config_file = None
        self.update_timeout = update_timeout
        self.configuration, self.enabled_servers = self._parse_proxy_config(config_str)
        self.server_pool = [client_class(server, server_timeout) for server in self.get_enabled_servers()]
        self.default_server = self.server_pool[0]
        self.executor = ThreadPoolExecutor(len(self.server_pool))

        self.permanent_instances_file = self.get_config_folder() + "/permanent_instances.json"
        self.permanent_instances = {}
        self.permanent_instances_manager_timer = None
        try:
            with open(self.permanent_instances_file ) as data_file:
                self.set_permanent_instances(json.load(data_file))
        except:
            _logger.warning("Error loading permanent instances: " + str(sys.exc_info()[1]))

    def resolve_server(self, server_index_or_name):
        for s in self.server_pool:
            compact_name = s.address.replace("http", "").replace("/", "").replace(":", "")
            if compact_name == server_index_or_name:
                return s
        return self.server_pool[int(server_index_or_name)]

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
                    "instances": instances
                    }
            for key in ["version", "cpu", "memory", "tx", "rx"]:
                ret[key] = [(info[server].get(key) if info[server] else None) for server in info] if info else []
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

        @app.get(api_root_address + "/info/<params:re:.*>")
        def get_nested_info(params):
            info = self.get_info()
            param_list = params.split('/')
            for p in param_list:
                try:
                    if p == 'keys':
                        info = list(info.keys())
                    elif p == 'values':
                        info = list(info.values())
                    else:
                        index = int(p)
                        if type(info) is list:
                            info = info[index]
                        elif type(info) is dict:
                            info = info[list(info.keys())[index]]
                        else:
                            raise Exception()
                except:
                    info = info[p]
            return {"state": "ok",
                    "status": "Nested information.",
                    "path": params,
                    "ret": info}

        @app.delete(api_root_address + "/<instance_name>")
        def stop_instance(instance_name):
            """
            Stop a specific instance.
            :param instance_name: Name of the instance.
            """
            self.stop_instance(instance_name)

            return {"state": "ok",
                    "status": "Instance '%s' stopped." % instance_name}

        @app.delete(api_root_address + "/server/<server_index_or_name>")
        def stop_all_server_instances(server_index_or_name):
            """
            Stop a specific camera.
            :param server_index_or_name
            """
            server = self.resolve_server(server_index_or_name)
            self.stop_all_server_instances(server)
            return {"state": "ok",
                    "status": "All instances stopped in '%s'." % server.get_address()}

        @app.delete(api_root_address + "/<instance_name>/del")
        def delete_instance(instance_name):
            """
            Stop and delete a specific instance.
            :param instance_name: Name of the camera.
            """
            self.stop_instance(instance_name, delete_instance=True)

            return {"state": "ok",
                    "status": "Instance '%s' stopped." % instance_name}

        @app.get(api_root_address + "/server/logs/<server_index_or_name>")
        def get_server_logs(server_index_or_name):
            """
            Return the list of logs
            :param server_index_or_name
            """
            response.content_type = 'application/json'
            server = self.resolve_server(server_index_or_name)
            logs = server.get_logs(txt=False)
            logs = list(logs) if logs else []
            return {"state": "ok",
                    "status": "Server logs.",
                    "logs": logs
                    }

        @app.get(api_root_address + "/server/logs/<server_index_or_name>/txt")
        def get_server_logs_txt(server_index_or_name):
            """
            Return the list of logs
            :param server_index
            """
            response.content_type = 'text/plain'
            server = self.resolve_server(server_index_or_name)
            logs = server.get_logs(txt=True)
            return logs

        @app.get(api_root_address + "/server/instance/logs/<server_index_or_name>/<instance_name>")
        def get_instance_logs(server_index_or_name, instance_name):
            """
            Return the list of logs
            :param server_index_or_name
            """
            response.content_type = 'application/json'
            server = self.resolve_server(server_index_or_name)
            logs = server.get_instance_logs(instance_name, txt=False)
            logs = list(logs) if logs else []
            return {"state": "ok",
                    "status": "Server logs.",
                    "logs": logs
                    }

        @app.get(api_root_address + "/server/instance/logs/<server_index_or_name>/<instance_name>/txt")
        def get_instance_logs_txt(server_index_or_name, instance_name):
            """
            Return the list of logs
            :param server_index
            """
            response.content_type = 'text/plain'
            server = self.resolve_server(server_index_or_name)
            logs = server.get_instance_logs(instance_name, txt=True)
            return logs


        @app.get(api_root_address + '/config')
        def get_config():
            """
            Get proxy config.
            :return: Configuration.
            """

            return {"state": "ok",
                    "status": "Proxy configuration retrieved.",
                    "config": self.configuration}

        @app.post(api_root_address + '/config')
        def set_config():
            """
            Set the proxy config
            :return: New config.
            """
            new_config = request.json
            _logger.info("Setting proxy config: %s" % new_config)
            config_str = json.dumps(new_config, sort_keys=True, indent=4,) #Check if can serialize first

            self.configuration = new_config

            if self.config_file:
                with open(self.config_file, "w") as text_file:
                    text_file.write(config_str)
            return {"state": "ok",
                    "status": "Proxy configuration  saved.",
                    "config": self.configuration}

        @app.get(api_root_address + '/permanent')
        def get_permanent_instances():
            """
            Get list of permanent instances.
            :return: List.
            """
            return {"state": "ok",
                    "status": "Proxy configuration retrieved.",
                    "permanent_instances": self.permanent_instances}

        @app.get(api_root_address + '/config_names')
        def get_config_names():
            """
            Get list of configuration names.
            :return: List.
            """
            return {"state": "ok",
                    "status": "Configuration names retrieved.",
                    "config_names": self.get_config_names()}

        @app.get(api_root_address + '/<name>/config')
        def get_named_config(name):
            """
            Get instance  configuration.
            :param name: Name of the instance to retrieve the config for.
            :return: Config.
            """
            camera_config = self.config_manager.config_provider.get_config(name, True)

            return {"state": "ok",
                    "status": "%s configuration retrieved." % name,
                    "config": camera_config}

        @app.post(api_root_address + '/<name>/config')
        def set_named_config(name):
            """
            Set an instance  config.
            :param name: Name of the instnce to change the config for.
            :return: New config.
            """
            new_config = request.json
            _logger.info("Setting '%s' config: %s" % (name, str(new_config)))
            self.config_manager.config_provider.save_config(name, new_config)

            return {"state": "ok",
                    "status": "%s configuration saved." % name,
                    "config": new_config}

        @app.delete(api_root_address + '/<name>/config')
        def delete_named_config(name):
            """
            Delete instance  configuration.
            :param name: Name of the configuration to delete
            :return: Config.
            """
            self.config_manager.config_provider.delete_config(name)

            return {"state": "ok",
                    "status": "%s configuration deleted." % name}

        @app.post(api_root_address + '/permanent')
        def set_permanent_instances():
            """
            Set list of permanent instances.
            :return: List.
            """
            _logger.info("Setting permanent instances: %s" % request.json)

            self.set_permanent_instances(request.json)

            return {"state": "ok",
                    "status": "Proxy configuration  saved.",
                    "permanent_instances": self.permanent_instances}

        @app.get(api_root_address + '/version')
        def get_version():
            """
            Get proxy versiom.
            :return: Version.
            """

            return {"state": "ok",
                    "status": "Version",
                    "version":  __VERSION__}

        @app.get(api_root_address + "/heartbeat")
        def get_heartbeat_info():
            """
            Return the proxy heartbeat info.
            """
            return {"state": "ok", "status": "Heartbeat", "version":  __VERSION__, "nodes":self.get_status()}


    def _exists_file(self, folder, file):
        return os.path.isfile(folder + "/" + file)

    def _get_root(self):
        folders = [self.get_config_folder()+"/www", os.path.dirname(__file__)]
        for folder in folders:
            if os.path.isfile(folder + "/index.html"):
                return folder

        msg = "Cannot locate index.html in " + str(folders)
        _logger.warning(msg)
        raise Exception(msg)

    def register_management_page(self, app):
        @app.route('/')
        def home():
            return static_file("index.html", self._get_root())


        @app.route('/utils.js')
        def utils():
            return static_file("utils.js", self._get_root())


    def get_status(self):
        def task(server):
            try:
                instances = server.get_server_info(
                    timeout = self.update_timeout if self.update_timeout else config.DEFAULT_SERVER_INFO_TIMEOUT)['active_instances']
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
                    info = server.get_server_info(
                        timeout = self.update_timeout if self.update_timeout else config.DEFAULT_SERVER_INFO_TIMEOUT)
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

    def get_enabled_servers(self):
        return self.enabled_servers

    def get_fixed_server(self, name):
        for server in self.get_enabled_servers():
            try:
                for entry in self.configuration[server]["instances"]:
                    if not entry.startswith('#'):
                        if ':' in entry:
                            [instance, port] = entry.split(":")
                        else:
                            instance, port = entry, None
                        if name == instance.strip():
                            return self.get_server_from_address(server), port
            except:
                pass
        return None, None

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

    def get_server_load(self, server, status=None):
        if status is None: status = self.get_status()
        if server in self.server_pool:
            try:
                return len(status[server.get_address()])
            except:
                pass
        return 1000

    def _get_free_server(self, servers, status=None):
        load = self.get_load(status)
        for i in range(len(self.server_pool)):
            name = self.server_pool[i].get_address()
            try:
                if self.configuration[name]["expanding"] == False:
                    load[i] = 1000
            except:
                pass
        m = min(load)
        if m >= 1000:
            raise Exception("No available server")
        return self.server_pool[load.index(m)]

    def get_free_server(self, instance_name=None, status=None):
        load = self.get_load(status)
        for i in range(len(self.server_pool)):
            name = self.server_pool[i].get_address()
            try:
                if self.configuration[name]["expanding"] == False:
                    load[i] = 1000
            except:
                pass
        m = min(load)
        if m >= 1000:
            raise Exception("No available server")
        return self.server_pool[load.index(m)]

    def get_source(self):
        server_address = request.environ.get('HTTP_X_FORWARDED_FOR') or request.environ.get('REMOTE_ADDR')
        server_name = socket.gethostbyaddr(server_address)[0]
        server_prefix = server_name.split(".")[0].lower() if server_name else None
        return server_name, server_prefix, server_address


    def get_request_server(self, status=None):
        try:
            servers = []
            load = self.get_load(status)
            loads = []
            server_name, server_prefix, server_address = self.get_source()

            for i in range(len(self.server_pool)):
                if load[i]<1000:
                    name = self.server_pool[i].get_address()
                    host, port = get_host_port_from_stream_address(name)
                    host_prefix= host.split(".")[0].lower() if host else None
                    local_host_name = socket.gethostname()
                    local_host_prefix= local_host_name.split(".")[0].lower() if local_host_name else None

                    if server_address == "127.0.0.1":
                        if (host =="127.0.0.1") or (host_prefix in (local_host_prefix, "localhost")):
                            servers.append(self.server_pool[i])
                            loads.append(load[i])
                    elif server_prefix == host_prefix:
                        servers.append(self.server_pool[i])
                        loads.append(load[i])
                    elif server_address == host:
                        servers.append(self.server_pool[i])
                        loads.append(load[i])

            #Balancing between servers in same machine to pass manager test
            if len(servers) > 0:
                m = min(loads)
                return servers[loads.index(m)]
        except Exception as e:
            _logger.warning('Failed to identify request origin: '+ str(e))
        return None

    def get_info(self, instances_only=False):
        info = self.get_servers_info()
        urls = info.keys()
        ret = {'active_instances': {}}
        servers_info = {}
        try:
            for server in urls:
                if info[server] != None:
                    active_instances = info[server].get('active_instances')
                    if active_instances is not None:
                        for k in active_instances.keys():
                            active_instances[k]["host"] = server
                        ret['active_instances'].update(active_instances)

            status = {server: (info[server]['active_instances'] if info[server] else None) for server in info}
            load = self.get_load(status)
            if instances_only:
                instances = []
                for server, status in status.items():
                    if status:
                        instances.extend(list(status.keys()))
                return instances
            i=0
            for server in urls:
                server_info = {}
                try:
                    server_info["instances"] = list(status[server].keys())
                    server_info["load"] =load[i]
                    for key in ["version", "cpu", "memory", "tx", "rx"]:
                        server_info[key] = info[server].get(key) if info[server] else None
                except:
                    server_info["instances"] = []
                    server_info["load"] = None
                    for key in ["version", "cpu", "memory", "tx", "rx"]:
                        server_info[key] = None
                servers_info[server] = server_info
                i = i + 1
        except Exception as e:
            _logger.warning("Error getting proxy info: " + str(sys.exc_info()[1]))
        ret['servers'] = servers_info
        return ret

    def get_instance_stream(self, instance_name):
        status = self.get_status()
        server = self.get_server(instance_name, status)
        port = None
        if server is None:
            server, port = self.get_fixed_server(instance_name)
            if server is None:
                server = self.get_request_server(status)
                if server is None:
                    server = self.get_free_server(instance_name, status)
                    _logger.info("Creating stream to %s at %s" % (instance_name, server.get_address()))
                else:
                    _logger.info("Creating stream to %s at request server %s Request: %s" % (instance_name, server.get_address(), str(self.get_source())))
            else:
                _logger.info("Creating fixed stream to %s at %s" % (instance_name, server.get_address()))
            self.on_creating_server_stream(server, instance_name, port)
        else:
            _logger.info("Connecting to stream %s at %s" % (instance_name, server.get_address()))
        return server.get_instance_stream(instance_name)

    def on_creating_server_stream(self, server, instance_name, port):
        pass

    def stop_all_instances(self):
        _logger.info("Stopping all")
        for server in self.server_pool:
            try:
                server.stop_all_instances()
            except:
                pass

    def stop_all_server_instances(self, server):
        _logger.info("Stopping all instances at %s" % server.get_address())
        try:
            server.stop_all_instances()
        except:
            pass

    def stop_instance(self, instance_name, delete_instance=False):
        status = self.get_status()
        server = self.get_server(instance_name, status)
        if server is not None:
            _logger.info("Stopping %s at %s" % (instance_name, server.get_address()))
            if delete_instance:
                server.delete_instance(instance_name)
            else:
                server.stop_instance(instance_name)

    def get_config_provider(self):
        return self.config_manager.config_provider

    def get_config_folder(self):
        try:
            return self.get_config_provider().config_folder
        except:
            return None

    def _parse_proxy_config(self, config_str):
        config_base = self.get_config_folder()
        # Server config in JSON file
        if not config_str:
            if config_base:
                self.config_file = config_base + "/servers.json"
                with open(self.config_file ) as data_file:
                    configuration = json.load(data_file)
            else:
                configuration = {}
        else:
            config_str = config_str.strip()
            # json
            if config_str.startswith("{"):
                configuration = json.loads(config_str)
            elif os.path.isfile(config_str):
                self.config_file = config_str
                with open(self.config_file) as data_file:
                    configuration = json.load(data_file)
            else:
                configuration = {}
                for server in [s.strip() for s in config_str.split(",")]:
                    configuration[server] = {"expanding": True}

        enabled_servers = []
        for server in configuration.keys():
            if configuration[server].get("enabled", True):
                enabled_servers.append(server)

        return configuration, enabled_servers

    def set_permanent_instances(self, permanent_instances):
        _logger.info("Setting permanent instances: " + str(permanent_instances))
        instances = self.config_manager.config_provider.get_available_configs()
        for name in list(permanent_instances.keys()):
            instance=permanent_instances[name]
            if not instance in instances:
                _logger.warning("Deleting non-existent permanent pipeline: " + str(instance))
                del permanent_instances[name]
        # Check if can serialize first
        permanent_instances_str = json.dumps(permanent_instances, sort_keys=True, indent=4, )

        with open(self.permanent_instances_file, "w") as text_file:
            text_file.write(permanent_instances_str)
        former = self.permanent_instances
        self.permanent_instances = permanent_instances

        if permanent_instances and not former:
            self.start_permanent_instances_manager()
        for name,instance in former.items():
            if not (name,instance) in permanent_instances.items():
                try:
                    self.stop_permanent_instance(instance,name)
                except:
                    _logger.warning("Error stopping permanent instance " + name + ": " + str(sys.exc_info()[1]))
        try:
            instances = self.get_info()['active_instances']
        except:
            instances = {}
        for name,instance in permanent_instances.items():
            if not name.startswith('#'):
                if not instance:
                    instance = name
                if name in instances.keys():
                    _logger.info("Permanent instance " + name + " is already running")
                else:
                    if not (instance,name) in former.items():
                        try:
                            self.start_permanent_instance(instance,name)
                        except:
                            _logger.warning("Error starting permanent instance " + name + ": " + str(sys.exc_info()[1]))
        if former and not permanent_instances:
            self.stop_permanent_instances_manager()

    def start_permanent_instances_manager(self):
        _logger.info("Starting permanent instance manager")

        self.schedule_timer()

    def schedule_timer(self):
        if self.permanent_instances_manager_timer:
            self.permanent_instances_manager_timer.cancel()
        self.permanent_instances_manager_timer = Timer(10, self.manage_permanent_instances)
        self.permanent_instances_manager_timer.daemon = True
        self.permanent_instances_manager_timer.start()

    def manage_permanent_instances(self):
        _logger.debug("Managing permanent instances")
        try:
            info = self.get_info()
            instances = info['active_instances']
            for name, instance in self.permanent_instances.items():
                if not name.startswith('#'):
                    if not instance:
                        instance = name
                    if not name in instances.keys():
                        try:
                            _logger.info("Instance not active: %s name: %s" % (instance, name))
                            self.start_permanent_instance(instance, name)
                        except:
                            _logger.warning("Error starting permanent instance " + instance + ": " + str(sys.exc_info()[1]))
        except:
            _logger.warning("Error managing permanent instances: " + str(sys.exc_info()[1]))
        self.schedule_timer()


    def stop_permanent_instances_manager(self):
        _logger.info("Stopping permanent instance manager")

        if self.permanent_instances_manager_timer:
            self.permanent_instances_manager_timer.cancel()

    def start_permanent_instance(self, instance, name):
        _logger.info("Starting permanent instance: %s name: %s" % (instance,name))
        raise Exception("Not implemented")

    def stop_permanent_instance(self, instance, name):
        _logger.info("Stopping permanent instance of %s: %s" % (instance,name))
        self.stop_instance(name)

    def is_permanent_instance(self, name):
        return not self.get_permanent_instance(name) is None


    def get_permanent_instance(self, name):
        for _name, instance in self.permanent_instances.items():
            if name == _name:
                if not _name.startswith('#'):
                    return instance
        return None

    def get_config_names(self):
        return []
