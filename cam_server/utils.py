from itertools import cycle
import logging
from logging import getLogger
from mflow.tools import ConnectionCountMonitor
from cam_server_client.utils import get_host_port_from_stream_address
import os
import collections
from bottle import response

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


def sum_images(image, accumulator_image):

    if accumulator_image is None:
        accumulator_image = numpy.array(image).astype(dtype="uint64")
    else:
        accumulator_image += image

    return accumulator_image


def get_clients(sender):
    if sender and sender.stream:
        for m in sender.stream._socket_monitors:
            if type(m) == ConnectionCountMonitor:
                return m.client_counter
    return 0


def set_statistics(statistics, sender, total_bytes, frame_count, frame_shape = None):
    now = time.time()
    timespan = now - statistics.timestamp
    statistics.update_timestamp = time.localtime()
    statistics.total_bytes = total_bytes
    statistics._frame_count = statistics._frame_count + frame_count
    if timespan > 1.0:
        received_bytes = total_bytes - statistics._last_proc_total_bytes
        statistics._last_proc_total_bytes = total_bytes
        statistics.clients = get_clients(sender)
        statistics.throughput = (received_bytes / timespan) if (timespan > 0) else None
        statistics.frame_rate = (statistics._frame_count / timespan) if (timespan > 0) else None
        statistics.frame_shape = frame_shape
        statistics._frame_count = 0
        statistics.timestamp = now
        if psutil and statistics._process:
            statistics.cpu = statistics._process.cpu_percent()
            statistics.memory = statistics._process.memory_info().rss
        else:
            statistics.cpu = None
            statistics.memory = None



def init_statistics(statistics):
    statistics.update_timestamp = None
    statistics.clients = 0
    statistics.total_bytes = 0
    statistics.throughput = 0
    statistics.frame_rate = 0
    statistics.frame_shape = None
    statistics.pid = os.getpid()
    statistics.cpu = 0
    statistics.memory = 0
    statistics.timestamp = time.time()
    if psutil:
        statistics._process = psutil.Process(os.getpid())
    statistics._frame_count = 0
    statistics._last_proc_total_bytes = 0


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


def initialize_api_logger(level = None, maxlen = 1000):
    global _api_logger, _api_log_buffer
    _api_logger = logging.getLogger("cam_server")
    _api_logger.setLevel(level if level else "INFO")
    _api_log_capture_string = None
    _api_log_buffer = collections.deque(maxlen=maxlen)
    handler = MyHandler()
    handler.setLevel(level if level else "INFO")
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    _api_logger.addHandler(handler)


def get_api_logs():
    return _api_log_buffer


def register_logs_rest_interface(app, api_root_address):
    _logger.warning("Start logging")

    @app.get(api_root_address)
    def get_logs():
        """
        Return the list of logs
        :return:
        """
        response.content_type = 'application/json'
        logs = get_api_logs()
        logs = list(logs) if logs else []
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
        logs = ("\n".join([" - ".join(log) for log in get_api_logs()]) ) if logs else ""
        return logs


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
