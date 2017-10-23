from logging import getLogger

import epics
from bsread import PULL, Source
from cam_server import config

from cam_server.camera.source.epics import CameraEpics
from cam_server.utils import get_host_port_from_stream_address

_logger = getLogger(__name__)


class CameraBsread(CameraEpics):
    def __init__(self, camera_config):
        super(CameraBsread, self).__init__(camera_config)
        self.bsread_stream_address = None
        self.bsread_source = None

    def _collect_camera_settings(self):
        super(CameraBsread, self)._collect_camera_settings()

        bsread_source_pv = self.camera_config.get_source() + config.EPICS_PV_SUFFIX_STREAM_ADDRESS

        _logger.debug("Checking camera bsread stream address '%s' PV.", bsread_source_pv)

        bsread_source = epics.PV(bsread_source_pv)

        self.bsread_stream_address = bsread_source.get(timeout=config.EPICS_TIMEOUT_GET)

        if not self.bsread_stream_address:
            raise RuntimeError("Could not fetch bsread stream address for cam_server:{}".format(
                self.camera_config.get_source()))

        bsread_source_pv.disconnect()

    def connect(self):

        self.verify_camera_online()
        self._collect_camera_settings()

        source_host, source_port = get_host_port_from_stream_address(self.bsread_stream_address)

        self.bsread_source = Source(host=source_host, port=source_port, mode=PULL)

        self.bsread_source.connect()

        return self.bsread_source

    def disconnect(self):
        self.bsread_source.disconnect()

    def __enter__(self):
        bsread_source = self.connect()
        return bsread_source

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

