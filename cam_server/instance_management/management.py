import multiprocessing
import time
from logging import getLogger
from cam_server import __VERSION__

from epics.multiproc import CAProcess as Process

from cam_server import config
from cam_server.utils import get_port_generator

try:
    import psutil
except:
    psutil = None

_logger = getLogger(__name__)


class InstanceManager(object):
    def __init__(self, port_range=None, auto_delete_stopped=False):
        self.instances = {}
        self._info_timestamp = None
        self._tx = None
        self._rx = None
        self._port_generator = get_port_generator(port_range) if (port_range is not None) else None
        self._used_ports = {}
        self._last_ports = {}
        self.auto_delete_stopped = auto_delete_stopped

    def get_next_available_port(self, instance_id, prefer_same_port = False):
        if self.auto_delete_stopped:
            self.delete_stopped_instances()

        if prefer_same_port:
            preferred_port = self._last_ports.get(instance_id)
            if preferred_port and (preferred_port not in self._used_ports):
                self._used_ports[preferred_port] = instance_id
                self._last_ports[instance_id] = preferred_port
                return preferred_port

        # Loop over all ports.
        for _ in range(*config.PIPELINE_STREAM_PORT_RANGE):
            candidate_port = next(self._port_generator)

            if candidate_port not in self._used_ports:
                self._used_ports[candidate_port] = instance_id
                self._last_ports[instance_id] = candidate_port
                return candidate_port

        raise Exception("All ports are used. Stop some instances before opening a new stream.")

    def delete_stopped_instance(self, instance_id):
        # If instance is present but not running, delete it.
        instance = self.instances.get(instance_id)
        if instance and not instance.is_running():
            _logger.info("Instance is present but not running: %s" % (instance_id,))
            port = instance.get_stream_port()
            self.delete_instance(instance_id)
            self._used_ports.pop(port, None)
            _logger.info("Instance deleted: %s" % (instance_id,))
        elif not instance:
            for port, id in self._used_ports.items():
                if id == instance_id:
                    self._used_ports.pop(port, None)
                    break


    def delete_stopped_instances(self):
        # Clean up any stopped instances.
        instance_ids = list(self._used_ports.values())
        for instance_id in instance_ids:
            self.delete_stopped_instance(instance_id)

    def is_stopped_instance(self, instance_id):
        instance = self.instances.get(instance_id)
        return instance and not instance.is_running()


    def get_info(self):
        """
        Return the instance manager info.
        :return: Dictionary with the info.
        """
        info = { "version": __VERSION__,
                 "active_instances": dict((instance.get_instance_id(), instance.get_info())
                                         for instance in self.instances.values() if instance.is_running())}

        if psutil:
            info["cpu"] = psutil.cpu_percent()
            info["memory"] = psutil.virtual_memory().used
            net = psutil.net_io_counters()
            tx = net.bytes_sent
            rx = net.bytes_recv
            now = time.time()
            info["tx"] = None
            info["rx"] = None
            if self._info_timestamp:
                timespan = now - self._info_timestamp
                if timespan > 0:
                    info["tx"] = (tx - self._tx) / timespan
                    info["rx"] = (rx - self._rx) / timespan
            self._info_timestamp = now
            self._tx = tx
            self._rx = rx

        else:
            info["cpu"] = None
            info["memory"] = None
            info["tx"] = None
            info["rx"] = None
        return info

    def add_instance(self, instance_name, instance_wrapper):
        """
        Add instance to the list of instances.
        :param instance_name: Instance name to add.
        :param instance_wrapper: Instance wrapper.
        """
        self.instances[instance_name] = instance_wrapper

    def is_instance_present(self, instance_name):
        """
        Check if instance is already present in the instances pool.
        :param instance_name: Name to check.
        :return: True if instance is already present.
        """
        if self.auto_delete_stopped:
            self.delete_stopped_instance(instance_name)
        return instance_name in self.instances

    def get_instance_stream_port(self, instance_name):
        if instance_name in self.instances:
            self.instances[instance_name].stream_port

    def _get_instance(self, instance_name):
        if instance_name not in self.instances:
            raise ValueError("Instance '%s' does not exist." % instance_name)
        return self.instances[instance_name]

    def get_instance(self, instance_name):
        """
        Retrieve the requested instance.
        :param instance_name: Name od the instance to return.
        :return:
        """
        if self.auto_delete_stopped:
            self.delete_stopped_instance(instance_name)
        return self._get_instance(instance_name)

    def get_instance_exit_code(self, instance_name):
        instance = self._get_instance(instance_name)
        if not instance.process:
            raise ValueError("Instance '%s' process not created." % instance_name)
        return instance.process.exitcode


    def start_instance(self, instance_name):
        """
        Start the instance.
        :param instance_name: Instance to start.
        """
        _logger.info("Starting instance '%s'." % instance_name)
        if instance_name in self.instances:
            instance = self.instances[instance_name]
            if not instance.is_running():
                instance.start()
            else:
                _logger.info("Instance '%s' is already running." % instance_name)
        else:
            raise ValueError("Instance '%s' does not exist." % instance_name)

    def stop_instance(self, instance_name):
        """
        Terminate the instance of the specified name.
        :param instance_name: Name of the instance to stop.
        """
        _logger.info("Stopping instance '%s'." % instance_name)

        if instance_name in self.instances:
            self.instances[instance_name].stop()
        if self.auto_delete_stopped:
            self.delete_stopped_instance(instance_name)

    def stop_all_instances(self):
        """
        Terminate all the instances.
        :return:
        """
        _logger.info("Stopping all instances.")

        for instance_id in list(self.instances.keys()):
            self.stop_instance(instance_id)


    def delete_instance(self, instance_name):
        self._get_instance(instance_name)
        del self.instances[instance_name]


class InstanceWrapper:
    def __init__(self, instance_name, process_function, stream_port, *args):

        self.instance_name = instance_name
        self.process_function = process_function
        self.stream_port = stream_port
        # Arguments to pass to the process function.
        self.args = args

        self.process = None
        self.manager = multiprocessing.Manager()

        self.stop_event = multiprocessing.Event()
        # The initial value of the stop event is set -> when the process starts, it un-sets it to signal the start.
        self.stop_event.set()

        self.statistics = self.manager.Namespace()
        self.statistics.total_bytes = 0
        self.statistics.clients = 0
        self.statistics.update_timestamp = None
        self.statistics.throughput = 0
        self.statistics.frame_rate = 0
        self.statistics.frame_shape = None
        self.statistics.timestamp = 0
        self.statistics.pid = ""
        self.statistics.cpu = 0
        self.statistics.memory = 0
        self.statistics._process = None
        self.statistics._frame_count = 0


        self.parameter_queue = multiprocessing.Queue()

        self.last_start_time = None

    def start(self):
        if self.process and self.process.is_alive():
            _logger.info("Instance '%s' already running." % self.instance_name)
            return

        self.stop_event.set()

        self.process = Process(target=self.process_function,
                               args=(self.stop_event, self.statistics, self.parameter_queue,
                                     *self.args))
        self.process.start()

        # Wait for the processor to clear the flag - indication that the process is ready.
        start_timestamp = time.time()
        error_message = None
        while self.stop_event.is_set():
            time.sleep(config.PROCESS_POLL_INTERVAL)
            if not self.process.is_alive():
                error_message = "'%s' instance  terminated. See logs." % self.instance_name
            # Check if the timeout has already elapsed.
            if time.time() - start_timestamp > config.PROCESS_COMMUNICATION_TIMEOUT:
                self.process.terminate()
                error_message = "Could not start the '%s' instance in time. Terminated. See logs." %  self.instance_name
            if error_message:
                _logger.error(error_message)
                raise Exception(error_message)

        self.last_start_time = time.strftime(config.TIME_FORMAT)

    def stop(self):
        if not self.process:
            _logger.info("Instance '%s' already stopped." % self.instance_name)
            return

        self.stop_event.set()

        # Wait for the processor to clear the flag - indication that the process is ready.
        start_timestamp = time.time()

        while self.process.is_alive():
            time.sleep(config.PROCESS_POLL_INTERVAL)
            # Check if the timeout has already elapsed.
            if time.time() - start_timestamp > config.PROCESS_COMMUNICATION_TIMEOUT:
                _logger.warning("Could not stop the '%s' instance in time. Terminated." % self.instance_name)
                break

        # Kill process - no-op in case process already terminated
        self.process.terminate()
        self.process = None

    def wait(self):
        self.process.join()

    def set_parameter(self, parameters):
        self.parameter_queue.put(parameters)

    def is_running(self):
        return self.process and self.process.is_alive()

    def get_instance_id(self):
        return self.instance_name

    def get_statistics(self):
        ret = {"total_bytes": self.statistics.total_bytes,
                "clients": self.statistics.clients,
                "throughput": self.statistics.throughput,
                "time": "" if not self.statistics.update_timestamp else time.strftime("%H:%M:%S", self.statistics.update_timestamp),
                "frame_rate": self.statistics.frame_rate,
                "pid": str(self.statistics.pid),
                "cpu": self.statistics.cpu,
                "memory": self.statistics.memory,
                }
        if self.statistics.frame_shape:
            ret["frame_shape"] = self.statistics.frame_shape
        return ret

    def get_stream_port(self):
        return self.stream_port
