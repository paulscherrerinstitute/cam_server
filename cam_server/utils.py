import collections
import logging
from itertools import cycle
from logging import getLogger

from bottle import response
from mflow.tools import ConnectionCountMonitor

try:
    import psutil
except:
    psutil = None

import os
import shutil
import time
import numpy
import ast
from bottle import ServerAdapter
import threading
import epics
import sys
import signal
from cam_server import config
from cam_server_client.utils import *

_logger = getLogger(__name__)


def update_pipeline_config(current_config, config_updates):
    def update_subsection(section_name):
        if config_updates.get(section_name) is not None:
            old_section = current_config.get(section_name)

            if old_section:
                old_section.update(config_updates.get(section_name))
                config_updates[section_name] = old_section

    update_subsection("image_good_region")
    update_subsection("image_slices")

    current_config.update(config_updates)

    return current_config


def update_camera_config(current_config, config_updates):
    if not config_updates:
        return config_updates

    def update_subsection(section_name):
        if config_updates.get(section_name) is not None:
            old_section = current_config.get(section_name)

            if old_section:
                old_section.update(config_updates.get(section_name))
                config_updates[section_name] = old_section

    update_subsection("camera_calibration")

    current_config.update(config_updates)

    return current_config

def get_port_generator(port_range):
    return cycle(iter(range(*port_range)))


def sum_images(image, accumulator_image, dtype="uint64"):

    if accumulator_image is None:
        accumulator_image = numpy.array(image).astype(dtype=dtype)
    else:
        accumulator_image += image.astype(dtype=dtype)

    return accumulator_image


def get_clients(sender):
    if sender and sender.stream:
        for m in sender.stream._socket_monitors:
            if type(m) == ConnectionCountMonitor:
                return m.client_counter
    return 0


def timestamp_as_float(timestamp):
    if timestamp is None:
        timestamp = time.time()
    # If you pass a tuple for the timestamp, use this tuple value directly.
    if isinstance(timestamp, tuple):
        return float(timestamp[0]) + (float(timestamp[1]) / 1e9)
    return timestamp

def on_message_sent():
    global msg_tx_counter
    if statistics is None:
        return
    statistics.tx_count = statistics.tx_count + 1
    if config.TELEMETRY_ENABLED:
        msg_tx_counter.add(1)

statistics = None

def update_statistics(sender, total_bytes_or_increment=0, frame_count=0, frame_shape = None, forwarder = None):
    if statistics is None:
        return
    now = time.time()
    timespan = now - statistics.timestamp
    try:
        statistics.header_changes = sender.header_changes
    except:
        statistics.header_changes = -1
    statistics.update_timestamp = time.localtime()
    if total_bytes_or_increment<=0:
        increment = -total_bytes_or_increment
        statistics.total_bytes = statistics.total_bytes+increment
    else:
        increment = total_bytes_or_increment - statistics.total_bytes
        statistics.total_bytes = total_bytes_or_increment
    statistics.rx_count = statistics.rx_count + frame_count
    statistics._frame_count = statistics._frame_count + frame_count
    if statistics.num_clients >= 0: #Multiprocessed
        statistics.clients = statistics.num_clients
    if timespan > 1.0:
        received_bytes = statistics.total_bytes - statistics._last_proc_total_bytes
        statistics._last_proc_total_bytes = statistics.total_bytes
        if statistics.num_clients < 0: #Not multiprocessed
            statistics.clients = get_clients(sender)
            if forwarder:
                statistics.clients = str(statistics.clients) + " + " + str(get_clients(forwarder))
        statistics.throughput = (received_bytes / timespan) if (timespan > 0) else None
        statistics.frame_shape = frame_shape
        statistics.frame_rate = (statistics._frame_count / timespan) if (timespan > 0) else 0
        statistics._frame_count = 0
        statistics.tx_rate = ((statistics.tx_count - statistics._tx_count) / timespan) if (timespan > 0) else 0
        statistics._tx_count = statistics.tx_count
        statistics.timestamp = now
        if psutil and statistics._process:
            statistics.cpu = statistics._process.cpu_percent()
            statistics.memory = statistics._process.memory_info().rss
        else:
            statistics.cpu = None
            statistics.memory = None
    if config.TELEMETRY_ENABLED:
        global msg_rx_counter, total_byte_counter
        msg_rx_counter.add(frame_count)
        total_byte_counter.add(increment)

def get_statistics():
    return statistics

def init_statistics(stats):
    global statistics
    statistics = stats
    statistics.update_timestamp = None
    statistics.num_clients = -1
    statistics.clients = 0
    statistics.total_bytes = 0
    statistics.throughput = 0
    statistics.frame_rate = 0
    statistics.rx_count = 0
    statistics.tx_count = 0
    statistics.header_changes = 0
    statistics._tx_count = 0
    statistics.frame_shape = None
    statistics.pid = os.getpid()
    statistics.cpu = 0
    statistics.memory = 0
    statistics.timestamp = time.time()
    if psutil:
        statistics._process = psutil.Process(os.getpid())
    statistics._frame_count = 0
    statistics._last_proc_total_bytes = 0

    if config.TELEMETRY_ENABLED:
        from opentelemetry.metrics import Observation, CallbackOptions
        from typing import Iterable
        global otel_get_meter, total_byte_counter, msg_rx_counter, msg_tx_counter, connected_clients, rx_rate, tx_rate, process_cpu, process_memory
        meter = otel_get_meter()
        total_byte_counter = meter.create_counter("total_byte_counter", description="Counter of received bytes")
        msg_rx_counter = meter.create_counter("msg_rx_count", description="Message rx counter")
        msg_tx_counter = meter.create_counter("msg_tx_count", description="Message tx counter")

        def connected_clients_func(options: CallbackOptions) -> Iterable[Observation]:
            yield Observation(statistics.num_clients, {})
        connected_clients = meter.create_observable_gauge("connected_clients", description="Number of connected clients", callbacks=[connected_clients_func])

        def rx_rate_func(options: CallbackOptions) -> Iterable[Observation]:
            yield Observation(statistics.throughput, {})
        rx_rate = meter.create_observable_gauge("rx_rate", description="Message rx counter", callbacks=[rx_rate_func])

        def tx_rate_func(options: CallbackOptions) -> Iterable[Observation]:
            yield Observation(statistics.throughput, {})
        tx_rate = meter.create_observable_gauge("tx_rate", description="Message rx counter", callbacks=[tx_rate_func])

        def process_cpu_func(options: CallbackOptions) -> Iterable[Observation]:
            yield Observation(statistics.cpu, {})
        process_cpu = meter.create_observable_gauge("process_cpu", description="Process CPU usage", callbacks=[process_cpu_func])

        def process_memory_func(options: CallbackOptions) -> Iterable[Observation]:
            yield Observation(statistics.cpu, {})
        process_memory= meter.create_observable_gauge("process_memory", description="Process Memory Usage", callbacks=[process_memory_func])


_api_logger = None
#_api_log_capture_string = None
_api_log_buffer = None





class MyHandler(logging.StreamHandler):
    def __init__(self):
        logging.StreamHandler.__init__(self)

    def emit(self, record):
        #_api_log_buffer.append(self.formatter.format(record))
        asctime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(record.created))
        _api_log_buffer.append([asctime, record.name, record.levelname, record.getMessage()])


def initialize_api_logger(level=config.APP_LOGGER_LEVEL, maxlen=config.APP_LOG_BUFFER_SIZE):
    global _api_logger, _api_log_buffer
    if level is None:
        level = config.APP_LOGGER_LEVEL
    _api_logger = logging.getLogger(config.APP_LOGGER)
    _api_logger.setLevel(level)
    _api_log_capture_string = None
    _api_log_buffer = collections.deque(maxlen=maxlen)
    handler = MyHandler()
    handler.setLevel(level)
    #handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    _api_logger.addHandler(handler)


def get_api_logs():
    global _api_logger, _api_log_buffer
    if _api_log_buffer:
        return list(_api_log_buffer)
    return []





def register_logs_rest_interface(app, api_root_address, instance_manager):
    _logger.warning("Start logging")

    def get_instance_logs(instance_name):
        try:
            instance = instance_manager.get_instance(instance_name)
            logs = list(instance.logs_queue)
            return logs
        except:
            return []


    @app.get(api_root_address)
    def get_logs():
        """
        Return the list of logs
        :return:
        """
        response.content_type = 'application/json'
        logs = get_api_logs()
        return {"state": "ok",
                "status": "Server logs.",
                "logs": logs
                }

    @app.get(api_root_address + "/txt")
    def get_logs_txt():
        """
        Return the list of logs
        :return:
        """
        response.content_type = 'text/plain'
        logs = get_api_logs()
        logs = "\n".join([" - ".join(log) for log in logs])
        return logs

    @app.get(api_root_address + "/instance/<instance_name>")
    def get_logs(instance_name):
        """
        Return the list of logs
        :return:
        """
        response.content_type = 'application/json'
        logs = get_instance_logs(instance_name)
        return {"state": "ok",
                "status": "Server logs.",
                "logs": logs
                }

    @app.get(api_root_address + "/instance/<instance_name>/txt")
    def get_logs_txt(instance_name):
        """
        Return the list of logs
        :return:
        """
        response.content_type = 'text/plain'
        logs = get_instance_logs(instance_name)
        logs = "\n".join([" - ".join(log) for log in logs])
        return logs



_instance_logs = None
_instance_logger = None
def setup_instance_logs(instance_logs, level=config.INSTANCE_LOGGER_LEVEL, maxlen=config.INSTANCE_LOG_BUFFER_SIZE):
    global _instance_logs, _instance_logger
    _instance_logs = instance_logs
    if _instance_logs is None:
        return
    if level is None:
            level = config.INSTANCE_LOGGER_LEVEL
    class MyHandler(logging.StreamHandler):
        def __init__(self):
            logging.StreamHandler.__init__(self)

        def emit(self, record):
            global _instance_logs
            asctime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(record.created))
            if len(_instance_logs) >= maxlen:
                del _instance_logs[0]
            _instance_logs.append([asctime, record.name, record.levelname, record.getMessage()])

    _instance_logger = logging.getLogger()
    _instance_logger.setLevel(level)
    handler = MyHandler()
    handler.setLevel(level)
    #handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    _instance_logger.addHandler(handler)



def remove(path, simulated=False):
    """
    Removes a file or directory.
    """
    if os.path.isdir(path):
        try:
            _logger.info("Removing folder: %s" % path)
            if not simulated:
                shutil.rmtree(path)
        except OSError:
            _logger.warning("Unable to remove folder: %s" % path)
    else:
        try:
            if os.path.exists(path):
                _logger.info("Removing file: %s" % path)
                if not simulated:
                    os.remove(path)
        except OSError:
            _logger.warning("Unable to remove file: %s" % path)


def cleanup(age_days, path, recursive=False, remove_folders=False, exceptions=[], simulated=False):
    """
    Removes files older than age_days.
    """
    _logger.info("Cleanup: %s - %d days old - recursive=%s remove_folders=%s exceptions=%s" %
                 (path, age_days, str(recursive), str(remove_folders), str(exceptions)))
    if not os.path.isdir(path):
        _logger.warning("Not a folder: %s" % path)
    else:
        seconds = time.time() - (age_days * 24 * 60 * 60)
        for root, dirs, files in os.walk(path, topdown=False):
            if recursive or (root == path):
                if not root[len(path)+1:] in exceptions:
                    for f in files:
                        p = os.path.join(root, f)
                        if os.stat(p).st_mtime <= seconds:
                            if not f in exceptions:
                                remove(p, simulated)
                    if remove_folders and (root != path):
                        if os.stat(root).st_mtime <= seconds:
                            #if not os.listdir(root):
                                remove(root, simulated)

def string_to_dict(str):
    if str:
        return ast.literal_eval(str)
    return{}



class MaxLenDict(collections.OrderedDict):
    def __init__(self, *args, **kwds):
        self.maxlen = kwds.pop("maxlen", None)
        collections.OrderedDict.__init__(self, *args, **kwds)
        self._check_maxlen()

    def __setitem__(self, key, value):
        collections.OrderedDict.__setitem__(self, key, value)
        self._check_maxlen()

    def _check_maxlen(self):
        if self.maxlen is not None:
            while len(self) > self.maxlen:
                self.popitem(last=False)


class CherryPyV9Server(ServerAdapter):
    def run(self, handler): # pragma: no cover
        from cheroot.wsgi import Server as WSGIServer
        self.options['bind_addr'] = (self.host, self.port)
        self.options['wsgi_app'] = handler

        certfile = self.options.get('certfile')
        if certfile:
            del self.options['certfile']
        keyfile = self.options.get('keyfile')
        if keyfile:
            del self.options['keyfile']

        server = WSGIServer(**self.options)
        if certfile:
            server.ssl_certificate = certfile
        if keyfile:
            server.ssl_private_key = keyfile

        try:
            server.start()
        finally:
            server.stop()

#Use this function to replace CherryPy adapter, as bottle has a bug starting CherryPy v>=9
#https://github.com/bottlepy/bottle/issues/934
#https://github.com/bottlepy/bottle/issues/975
def validate_web_server(web_server):
    if web_server=="cherrypy":
        return CherryPyV9Server
    return web_server


#EPICS utilities for threading
epics_lock = threading.RLock()

def create_pv(name, **args):
    with epics_lock:
        if epics.ca.current_context() is None:
            try:
                if epics.ca.initial_context is None:
                    _logger.info("Creating initial EPICS context for pid:" + str(os.getpid()) + " thread: " + str(
                        threading.get_ident()))
                    epics.ca.initialize_libca()
                else:
                    # TODO: using epics.ca.use_initial_context() generates a segmentation fault
                    # _logger.info("Using initial EPICS context for pid:" + str(os.getpid()) + " thread: " + str(threading.get_ident()))
                    # epics.ca.use_initial_context()
                    _logger.info("Creating EPICS context for pid:" + str(os.getpid()) + " thread: " + str(threading.get_ident()))
                    epics.ca.create_context()
            except:
                _logger.warning("Error creating PV context: " + str(sys.exc_info()[1]))
    return epics.PV(name, **args)

_thread_pvs = None

def create_thread_pv(pv_name, wait=True):
    global _thread_pvs, epics_lock
    with epics_lock:
        if _thread_pvs is None:
            _thread_pvs = collections.OrderedDict()
            epics.ca.clear_cache()
        if threading.get_ident() not in _thread_pvs.keys():
            _thread_pvs[threading.get_ident()] = {}
        if pv_name in _thread_pvs[threading.get_ident()].keys():
            return _thread_pvs[threading.get_ident()][pv_name]
        pv = create_pv(pv_name)
        _thread_pvs[threading.get_ident()][pv_name] = pv
    if wait:
        pv.wait_for_connection()
    return pv

def create_thread_pvs(pv_names):
    if isinstance(pv_names, str):
        pv_names = [pv_names]
    ret = []
    with epics_lock:
        if (_thread_pvs is not None) and (threading.get_ident() in _thread_pvs.keys()):
            pvs = _thread_pvs[threading.get_ident()]
            for name, pv in pvs.items():
                if not pv.connected:
                    pass # Must manage reconnection?
            return pvs.values()
    for name in pv_names:
        ret.append(create_thread_pv(name, False))
    for pv in ret:
        pv.wait_for_connection()
    return ret

def get_thread_pv(name):
    global _thread_pvs, epics_lock
    with epics_lock:
        if _thread_pvs is not None:
            if threading.get_ident() in _thread_pvs.keys():
                ret = _thread_pvs[threading.get_ident()].get(name)
                return ret

def get_thread_pvs():
    global _thread_pvs, epics_lock
    with epics_lock:
        if _thread_pvs is not None and threading.get_ident() in _thread_pvs.keys():
                return _thread_pvs[threading.get_ident()]
        return []

def remove_thread_pvs():
    with epics_lock:
        for pv in get_thread_pvs():
            try:
                pv.disconnect()
            except:
                pass
        if _thread_pvs is not None and threading.get_ident() in _thread_pvs.keys():
            del _thread_pvs[threading.get_ident()]

def reset():
    def thread_func():
        time.sleep(0.1)
        _logger.warning("Closing process")
        os.killpg(os.getpgid(os.getpid()), signal.SIGTERM)
        #os.kill(os.getpid(), signal.SIGTERM)
    th = threading.Thread(target=thread_func)
    th.start()


_thread_count = 0
_thread_semaphore = threading.Semaphore()
_thread_event = threading.Event()

def synchronise_threads(number_of_threads):
    global _thread_count, _thread_semaphore
    with _thread_semaphore:
        _thread_count += 1
        if _thread_count == number_of_threads:
            _thread_event.set()
    _thread_event.wait()


