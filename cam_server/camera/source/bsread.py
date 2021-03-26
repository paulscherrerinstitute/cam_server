from logging import getLogger

import epics
from bsread import PULL, Source
from cam_server import config

from cam_server.camera.source.epics import CameraEpics
from cam_server.utils import get_host_port_from_stream_address
from cam_server.camera.source.bsread_handler import Handler
from cam_server.camera.source.common import transform_image

_logger = getLogger(__name__)


class CameraBsread(CameraEpics):
    def __init__(self, camera_config):
        super(CameraBsread, self).__init__(camera_config)
        self.bsread_stream_address = None
        self.bsread_source = None

    def connect(self):
        self.verify_camera_online()
        self._collect_camera_settings()

    def _collect_camera_settings(self):
        super(CameraBsread, self)._collect_camera_settings()

        _logger.info("Collecting bsread camera settings.")

        bsread_source_pv = self.camera_config.get_source() + config.EPICS_PV_SUFFIX_STREAM_ADDRESS

        _logger.debug("Checking camera bsread stream address '%s' PV." % bsread_source_pv)

        self.bsread_stream_address = self.caget(bsread_source_pv)
        _logger.info("Got stream address: %s" % str(self.bsread_stream_address))
        if not self.bsread_stream_address:
            raise RuntimeError("Could not fetch bsread stream address for cam_server:{}".format(
                self.camera_config.get_source()))

    def get_stream(self, timeout=config.ZMQ_RECEIVE_TIMEOUT, data_change_callback=None):
        source_host, source_port = get_host_port_from_stream_address(self.bsread_stream_address)

        self.bsread_source = Source(host=source_host, port=source_port, mode=PULL,
                                    receive_timeout=timeout)
        self.bsread_source.handler = Handler(data_change_callback)
        return self.bsread_source


class CameraBsreadSim (CameraBsread):
        def __init__(self, camera_config, width=659, height=494, stream_address="tcp://0.0.0.0:9999"):
            super(CameraBsreadSim, self).__init__(camera_config)
            self.bsread_stream_address = stream_address

        def verify_camera_online(self):
            pass

        def _collect_camera_settings(self):
            try:
                #self.width_raw, self.height_raw  = 659, 494
                source_host, source_port = get_host_port_from_stream_address(self.bsread_stream_address)

                stream = Source(host=source_host, port=source_port, mode=PULL,receive_timeout=3000)
                stream.connect()
                data = stream.receive()
                image = data.data.data[self.camera_config.get_source() + config.EPICS_PV_SUFFIX_IMAGE].value
                if image is None:
                    self.height_raw, self.width_raw = 0,0
                else:
                    image = transform_image(image, self.camera_config)
                    self.height_raw, self.width_raw = image.shape
            except:
                raise RuntimeError("Could not fetch camera settings cam_server:{}".format(self.camera_config.get_source()))
            finally:
                stream.disconnect()
