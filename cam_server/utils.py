from itertools import cycle
from logging import getLogger
from mflow.tools import ConnectionCountMonitor
import os

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
    received_bytes = total_bytes - statistics.total_bytes
    now = time.time()
    timespan = now - statistics.timestamp
    statistics.clients = get_clients(sender)
    statistics.total_bytes = total_bytes
    statistics.throughput = (received_bytes / timespan) if (timespan > 0) else None
    statistics.frame_rate = (1.0 / timespan) if (timespan > 0) else None
    statistics.timestamp = now
    if psutil and statistics._process:
        if now - statistics.cpu_sampling_time > 1.0:
            statistics.cpu = statistics._process.cpu_percent()
            statistics.memory = statistics._process.memory_info().rss #Physical, Virtual = .vms
            statistics.cpu_sampling_time = now
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
    statistics.cpu_sampling_time = statistics.timestamp
    statistics._process = psutil.Process(os.getpid())