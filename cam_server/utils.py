from itertools import cycle
import logging
from logging import getLogger
from mflow.tools import ConnectionCountMonitor
import os
import collections
from bottle import response

try:
    import psutil
except:
    psutil = None

import time
import numpy

_logger = getLogger(__name__)


def get_host_port_from_stream_address(stream_address):
    source_host, source_port = stream_address.rsplit(":", maxsplit=1)
    source_host = source_host.split("//")[1]

    return source_host, int(source_port)


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
    for m in sender.stream._socket_monitors:
        if type(m) == ConnectionCountMonitor:
            return m.client_counter
    return 0


def set_statistics(statistics, sender, total_bytes):
    now = time.time()
    timespan = now - statistics.timestamp
    statistics._frame_count = statistics._frame_count + 1
    if timespan > 1.0:
        received_bytes = total_bytes - statistics.total_bytes
        statistics.clients = get_clients(sender)
        statistics.total_bytes = total_bytes
        statistics.throughput = (received_bytes / timespan) if (timespan > 0) else None
        statistics.frame_rate = (statistics._frame_count / timespan) if (timespan > 0) else None
        statistics._frame_count = 0
        statistics.timestamp = now
        if psutil and statistics._process:
            statistics.cpu = statistics._process.cpu_percent()
            statistics.memory = statistics._process.memory_info().rss
        else:
            statistics.cpu = None
            statistics.memory = None



def init_statistics(statistics):
    statistics.clients = 0
    statistics.total_bytes = 0
    statistics.throughput = 0
    statistics.frame_rate = 0
    statistics.pid = os.getpid()
    statistics.cpu = 0
    statistics.memory = 0
    statistics.timestamp = time.time()
    statistics._process = psutil.Process(os.getpid())
    statistics._frame_count = 0


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


