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

        _logger.info("Collecting bsread camera settings.")

        bsread_source_pv = self.camera_config.get_source() + config.EPICS_PV_SUFFIX_STREAM_ADDRESS

        _logger.debug("Checking camera bsread stream address '%s' PV." % bsread_source_pv)

        bsread_source = self.create_pv(bsread_source_pv)

        self.bsread_stream_address = bsread_source.get(timeout=config.EPICS_TIMEOUT)

        _logger.info("Got stream address: %s" % self.bsread_stream_address)

        if not self.bsread_stream_address:
            raise RuntimeError("Could not fetch bsread stream address for cam_server:{}".format(
                self.camera_config.get_source()))

        bsread_source.disconnect()

    def get_stream(self, timeout=config.ZMQ_RECEIVE_TIMEOUT):

        self.verify_camera_online()
        self._collect_camera_settings()

        source_host, source_port = get_host_port_from_stream_address(self.bsread_stream_address)

        self.bsread_source = Source(host=source_host, port=source_port, mode=PULL,
                                    receive_timeout=timeout)

        return self.bsread_source



class CameraBsreadSim (CameraBsread):
        def __init__(self, camera_config, width=659, height=494, stream_address="tcp://0.0.0.0:9999"):
            super(CameraBsreadSim, self).__init__(camera_config)
            self.width_raw = width
            self.height_raw = height
            self.bsread_stream_address = stream_address

        def verify_camera_online(self):
            pass

        def _collect_camera_settings(self):
            pass
