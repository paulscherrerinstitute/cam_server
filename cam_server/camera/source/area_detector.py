import sys
from logging import getLogger

import epics

from cam_server import config
from cam_server.camera.source.camera import Camera
from cam_server.camera.source.common import transform_image
from cam_server.utils import create_pv

_logger = getLogger(__name__)

class AreaDetector(Camera):

    def __init__(self, camera_config):
        """
        Create EPICS camera source.
        :param camera_config: Config of the camera.
        """
        super(AreaDetector, self).__init__(camera_config)

        self.prefix = self.camera_config.get_source()
        self.channel_ctrl = self.prefix + ":cam1"
        self.channel_data = self.prefix + ":image1"
        self.channel_image = None
        self.channel_width = None
        self.channel_height = None
        self.color_mode = None
        self.data_type = None
        self.image_counter = None
        self.counter = -1

    def caget(self, channel_name, timeout=config.EPICS_TIMEOUT, as_string=True):
        #channel = create_thread_pv(channel_name)
        channel = epics.PV(channel_name)
        try:
            ret = channel.get(timeout=timeout, as_string=as_string)
            if ret is None:
                raise Exception("Error getting channel %s for camera %s" % (channel_name, self.prefix))
            return ret
        finally:
            channel.disconnect()

    def connect_monitored_channel(self, name):
        ret = create_pv(name, auto_monitor=True)
        ret.wait_for_connection(config.EPICS_TIMEOUT)
        if not ret.connected:
            raise RuntimeError("Could not connect to: {}".format(ret.pvname))
        return ret

    def verify_camera_online(self):
        camera_init_pv = self.channel_ctrl + ":Acquire"
        try:
            if sys.platform == "darwin":
                return
            started = self.caget(camera_init_pv, as_string=False)
            # if started != 1:
            #    raise RuntimeError(("Camera with prefix %s is not started - Status %s" % (self.prefix, started)))
        except Exception as ex:
            raise RuntimeError(("Camera with prefix %s is offline - %s" % (self.prefix, str(ex))))


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
        if self._has_cache(self.channel_width):
            value = self.channel_width.get(timeout=config.EPICS_TIMEOUT, use_monitor=True)
            if not value:
                raise RuntimeError("Could not fetch width for cam_server:{}".format(self.prefix))
            self.width_raw = int(value)
        else:
            self.width_raw = int(self.caget(self.channel_data + ":ArraySize0_RBV"))

        if self._has_cache(self.channel_height):
            value = self.channel_height.get(timeout=config.EPICS_TIMEOUT, use_monitor=True)
            if not value:
                raise RuntimeError("Could not fetch height for cam_server:{}".format(self.prefix))
            self.height_raw = int(value)
        else:
            self.height_raw = int(self.caget(self.channel_data + ":ArraySize1_RBV"))

    def get_raw_geometry(self):
        if self.width_raw is None or self.height_raw is None:
            self.update_size_raw()
        return self.width_raw, self.height_raw

    def connect(self):
        self.verify_camera_online()

        # Connect image channel
        self.channel_image = self.connect_monitored_channel(self.channel_data + ":ArrayData")
        self.channel_width = self.connect_monitored_channel(self.channel_data + ":ArraySize0_RBV")
        self.channel_height = self.connect_monitored_channel(self.channel_data + ":ArraySize1_RBV")
        self.color_mode = self.connect_monitored_channel(self.channel_ctrl + ":ColorMode")
        self.data_type = self.connect_monitored_channel(self.channel_ctrl + ":DataType")
        self.image_counter = self.connect_monitored_channel(self.channel_ctrl + ":ArrayCounter_RBV")
        self.update_size_raw()


    def disconnect(self):
        self.clear_callbacks()
        for channel in [self.channel_image, self.channel_width , self.channel_height, self.color_mode, self.data_type, self.image_counter]:
            try:
                channel.disconnect()
            except:
                pass

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
                    _logger.info("Error getting image from camera %s: %s" % (self.prefix, sys.exc_info()[1]))
            else:
                _logger.debug("Null image read from camera %s" % (self.prefix))

        def _width_callback(value, timestamp, status, **kwargs):
            nonlocal shape_changed
            if (self.width_raw is not None) and (self.width_raw!=value):
                _logger.warning("Camera %s width changed: %d -> %d " % (self.prefix, self.width_raw, value))
                self.width_raw = int(value)
                shape_changed = True

        def _height_callback(value, timestamp, status, **kwargs):
            nonlocal shape_changed
            if (self.height_raw is not None) and (self.height_raw!=value):
                _logger.warning("Camera %s height changed: %d -> %d " % (self.prefix, self.height_raw, value))
                self.height_raw = int(value)
                shape_changed = True

        def _counter_callback(value, timestamp, status, **kwargs):
            self.counter = value

        self.channel_image.add_callback(_callback)
        self.channel_width.add_callback(_width_callback)
        self.channel_height.add_callback(_height_callback)
        self.image_counter.add_callback(_counter_callback)


    def _get_image(self, value, raw=False):

        if value is None:
            return None
        width, height = self.get_raw_geometry()

        size = width * height
        if value.size != size:
            if value.size < size:
                _logger.warning("Image array too small: %d -  shape: %dx%d [%s]." % (
                value.size, width,  height, self.prefix))
                return None
            else:
                value = value[:(size)]

        # Shape images
        value = value.reshape((height, width))


        if value.dtype == "uint8":
            value=value.astype("int16")
        elif value.dtype == "uint16":
            value=value.astype("int32")
        elif value.dtype == "uint32":
            value=value.astype("int64")
        # Return raw image without any corrections
        if raw:
            return value
        return transform_image(value, self.camera_config)

    def get_pulse_id(self):
        return self.image_counter.get(use_monitor=True)

    def get_image(self, raw=False):
        if self._has_cache(self.channel_image):
            # If we are already connected, just grab current image.
            value = self.channel_image.get(use_monitor=True)
        else:
            value = self.caget(self.channel_data + ":ArrayData", as_string=False)
        return self._get_image(value, raw=raw)

    def clear_callbacks(self):
        if self.channel_image:
            self.channel_image.clear_callbacks()
        if self.channel_width:
            self.channel_width.clear_callbacks()
        if self.channel_height:
            self.channel_height.clear_callbacks()
        if self.image_counter:
            self.image_counter.clear_callbacks()
