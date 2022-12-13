import sys

from logging import getLogger

from cam_server import config
from cam_server.camera.source.camera import Camera
from cam_server.camera.source.common import transform_image
from cam_server.utils import create_pv

_logger = getLogger(__name__)


class CameraEpics(Camera):

    def __init__(self, camera_config):
        """
        Create EPICS camera source.
        :param camera_config: Config of the camera.
        """
        super(CameraEpics, self).__init__(camera_config)
        self.channel_image = None
        self.channel_width = None
        self.channel_height = None


    def caget(self, channel_name, timeout=config.EPICS_TIMEOUT, as_string=True):
        channel = create_pv(channel_name)
        try:
            ret = channel.get(timeout=timeout, as_string=as_string)
            if ret is None:
                raise Exception("Error getting channel %s for camera %s" % (channel_name, self.camera_config.get_source()))
            return ret
        finally:
            channel.disconnect()

    def connect_monitored_channel(self, suffix):
        ret = create_pv(self.camera_config.get_source() + suffix, auto_monitor=True)
        ret.wait_for_connection(config.EPICS_TIMEOUT)
        if not ret.connected:
            raise RuntimeError("Could not connect to: {}".format(ret.pvname))
        return ret

    def verify_camera_online(self):
        camera_prefix = self.camera_config.get_source()
        camera_init_pv = camera_prefix + config.EPICS_PV_SUFFIX_STATUS
        try:
            channel_init_value = self.caget(camera_init_pv, as_string=True)
        except:
            raise RuntimeError(("Camera with prefix %s is offline" % (camera_prefix)))
        if channel_init_value != 'INIT':
            raise RuntimeError(("Camera with prefix %s is not initialized - Status %s" % (camera_prefix, channel_init_value)))

    def _has_cache(self, channel):
        if channel is not None:
            try:
                if channel._args['value'] is not None:
                    return True
            except:
                pass
        return False

    def update_size_raw(self):
        #If using channel.get without monitor there is an error creating the same channel in following processes
        #For every get with no cache the channel must be disconnected after access
        import threading
        if self._has_cache(self.channel_width):
            value = self.channel_width.get(timeout=config.EPICS_TIMEOUT, use_monitor=True)
            if not value:
                raise RuntimeError("Could not fetch width for cam_server:{}".format(self.camera_config.get_source()))
            self.width_raw = int(value)
        else:
            self.width_raw = int(self.caget(self.camera_config.get_source() + config.EPICS_PV_SUFFIX_WIDTH))

        if self._has_cache(self.channel_height):
            value = self.channel_height.get(timeout=config.EPICS_TIMEOUT, use_monitor=True)
            if not value:
                raise RuntimeError("Could not fetch height for cam_server:{}".format(self.camera_config.get_source()))
            self.height_raw = int(value)
        else:
            self.height_raw = int(self.caget(self.camera_config.get_source() + config.EPICS_PV_SUFFIX_HEIGHT))

    def _collect_camera_settings(self):
        self.update_size_raw()

    def connect(self):
        self.verify_camera_online()
        self._collect_camera_settings()
        # Connect image channel
        self.channel_image = self.connect_monitored_channel(config.EPICS_PV_SUFFIX_IMAGE)
        self.channel_width = self.connect_monitored_channel(config.EPICS_PV_SUFFIX_WIDTH)
        self.channel_height = self.connect_monitored_channel(config.EPICS_PV_SUFFIX_HEIGHT)

    def disconnect(self):
        self.clear_callbacks()
        if self.channel_image:
            try:
                self.channel_image.disconnect()
            finally:
                self.channel_image = None
        if self.channel_width:
            try:
                self.channel_width.disconnect()
            finally:
                self.channel_width = None
        if self.channel_height:
            try:
                self.channel_height.disconnect()
            finally:
                self.channel_height = None

    def add_callback(self, callback_function):
        shape_changed = False
        def _callback(value, timestamp, status, **kwargs):
            nonlocal shape_changed
            image = self._get_image(value)
            if image is not None:
                changed = shape_changed
                if changed:
                    shape_changed = False
                try:
                    callback_function(image, timestamp, changed)
                except:
                    _logger.info("Error getting image from camera %s: %s" % (self.camera_config.get_source(), sys.exc_info()[1]))
            else:
                _logger.debug("Null image read from camera %s" % (self.camera_config.get_source()))

        def _width_callback(value, timestamp, status, **kwargs):
            nonlocal shape_changed
            if (self.width_raw is not None) and (self.width_raw!=value):
                _logger.warning("Camera %s width changed: %d -> %d " % (self.camera_config.get_source(), self.width_raw, value))
                self.width_raw = int(value)
                shape_changed = True

        def _height_callback(value, timestamp, status, **kwargs):
            nonlocal shape_changed
            if (self.height_raw is not None) and (self.height_raw!=value):
                _logger.warning("Camera %s height changed: %d -> %d " % (self.camera_config.get_source(), self.height_raw, value))
                self.height_raw = int(value)
                shape_changed = True

        self.channel_image.add_callback(_callback)
        self.channel_width.add_callback(_width_callback)
        self.channel_height.add_callback(_height_callback)

    def get_raw_geometry(self):
        if self.width_raw is None or self.height_raw is None:
            self._collect_camera_settings()
        return self.width_raw, self.height_raw

    def _get_image(self, value, raw=False):

        if value is None:
            return None
        width, height = self.get_raw_geometry()

        size = width * height
        if value.size != size:
            if value.size < size:
                _logger.warning("Image array too small: %d -  shape: %dx%d [%s]." % (
                value.size, width,  height, self.camera_config.get_source()))
                return None
            else:
                value = value[:(size)]

        # Shape image
        value = value.reshape((height, width))

        # Return raw image without any corrections
        if raw:
            return value

        return transform_image(value, self.camera_config)

    def get_image(self, raw=False):
        if self._has_cache(self.channel_image):
            # If we are already connected, just grab current image.
            value = self.channel_image.get(use_monitor=True)
        else:
            value = self.caget(self.camera_config.get_source() + config.EPICS_PV_SUFFIX_IMAGE, as_string=False)
        return self._get_image(value, raw=raw)

    def clear_callbacks(self):
        if self.channel_image:
            self.channel_image.clear_callbacks()
        if self.channel_width:
            self.channel_width.clear_callbacks()
        if self.channel_height:
            self.channel_height.clear_callbacks()

