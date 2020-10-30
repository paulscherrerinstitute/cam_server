import sys
import epics
import numpy
import threading
import os

from logging import getLogger

from cam_server import config
from cam_server.camera.source.common import transform_image
from cam_server.pipeline.data_processing.functions import binning

_logger = getLogger(__name__)


class CameraEpics:

    def __init__(self, camera_config):
        """
        Create EPICS camera source.
        :param camera_config: Config of the camera.
        """
        self.camera_config = camera_config

        # Width and height of the raw image
        self.width_raw = None
        self.height_raw = None

        self.channel_image = None
        self.channel_width = None
        self.channel_height = None
        self.channel_creation_lock = threading.Lock()

    #Thread-safe pv creation
    def create_pv(self, name, **args):
        with self.channel_creation_lock:
            if epics.ca.current_context() is None:
                try:
                     if epics.ca.initial_context is None:
                         _logger.info("Creating initial EPICS context for pid:" + str(os.getpid()) + " thread: " + str(threading.get_ident()))
                         epics.ca.initialize_libca()
                     else:
                         # TODO: using epics.ca.use_initial_context() generates a segmentation fault
                         #_logger.info("Using initial EPICS context for pid:" + str(os.getpid()) + " thread: " + str(threading.get_ident()))
                         #epics.ca.use_initial_context()
                         _logger.info("Creating EPICS context for pid:" + str(os.getpid()) + " thread: " + str(threading.get_ident()))
                         epics.ca.create_context()
                except:
                    _logger.warning("Error creating PV context: " + str(sys.exc_info()[1]))
        return epics.PV(name, **args)

    def caget(self, channel_name, timeout=config.EPICS_TIMEOUT, as_string=True):
        channel = self.create_pv(channel_name)
        try:
            ret = channel.get(timeout=timeout, as_string=as_string)
            if ret is None:
                raise Exception("Error getting channel %s for camera %s" % (channel_name, self.camera_config.get_source()))
            return ret
        finally:
            channel.disconnect()

    def connect_monitored_channel(self, suffix):
        ret = self.create_pv(self.camera_config.get_source() + suffix, auto_monitor=True)
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

    def _collect_camera_settings(self):
        if self.channel_width is not None:
            value = self.channel_width.get(timeout=config.EPICS_TIMEOUT)
            if not value:
                raise RuntimeError("Could not fetch width for cam_server:{}".format(self.camera_config.get_source()))
            self.width_raw = int(value)
        else:
            self.width_raw = int(self.caget(self.camera_config.get_source() + config.EPICS_PV_SUFFIX_WIDTH))

        if self.channel_height is not None:
            value = self.channel_height.get(timeout=config.EPICS_TIMEOUT)
            if not value:
                raise RuntimeError("Could not fetch height for cam_server:{}".format(self.camera_config.get_source()))
            self.height_raw = int(value)
        else:
            self.height_raw = int(self.caget(self.camera_config.get_source() + config.EPICS_PV_SUFFIX_HEIGHT))

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

    def _get_image(self, value, raw=False):

        if value is None:
            return None

        size = self.width_raw * self.height_raw
        if value.size != size:
            if value.size < size:
                _logger.warning("Image array too small: %d -  shape: %dx%d [%s]." % (
                value.size, self.width_raw,  self.height_raw, self.camera_config.get_source()))
                return None
            else:
                value = value[:(size)]

        # Shape image
        value = value.reshape((self.height_raw, self.width_raw))

        # Return raw image without any corrections
        if raw:
            return value

        return transform_image(value, self.camera_config)

    def get_image(self, raw=False):
        # If we are not connected to the image channel, we have to do this first.
        if self.channel_image is None:
            self.connect()
            value = self.channel_image.get()
            self.disconnect()

        # If we are already connected, just grab the next image.
        else:
            value = self.channel_image.get()

        return self._get_image(value, raw=raw)

    def get_geometry(self):
        if self.width_raw is None or self.height_raw is None:
            self._collect_camera_settings()

        width, height = self.width_raw, self.height_raw
        if self.camera_config.parameters.get("binning_y"):
            height = int(height / self.camera_config.parameters.get("binning_y"))
        if self.camera_config.parameters.get("binning_x"):
            width = int(width / self.camera_config.parameters.get("binning_x"))

        rotate = self.camera_config.parameters["rotate"]
        if rotate == 1 or rotate == 3:
            # If rotating by 90 degree, height becomes width.
            return height, width
        else:
            return width, height

    def get_name(self):
        return self.camera_config.get_name()

    def clear_callbacks(self):
        if self.channel_image:
            self.channel_image.clear_callbacks()
        if self.channel_width:
            self.channel_width.clear_callbacks()
        if self.channel_height:
            self.channel_height.clear_callbacks()

    def get_x_y_axis(self):

        """
        Get x and y axis in nm based on calculated origin from the reference markers
        The coordinate system looks like this:
               +|
        +       |
        -----------------
                |       -
               -|
        Parameters
        ----------
        width       image with in pixel
        height      image height in pixel
        Returns
        -------
        (x_axis, y_axis)
        """

        calibration = self.camera_config.parameters["camera_calibration"]
        width, height = self.get_geometry()

        if not calibration:
            x_axis = numpy.linspace(0, width - 1, width, dtype='f')
            y_axis = numpy.linspace(0, height - 1, height, dtype='f')
        else:
            def _calculate_center():
                center_x = int(((lower_right_x - upper_left_x) / 2) + upper_left_x)
                center_y = int(((lower_right_y - upper_left_y) / 2) + upper_left_y)
                return center_x, center_y

            def _calculate_pixel_size():
                size_y = reference_marker_height / (lower_right_y - upper_left_y)
                size_y *= numpy.cos(vertical_camera_angle * numpy.pi / 180)

                size_x = reference_marker_width / (lower_right_x - upper_left_x)
                size_x *= numpy.cos(horizontal_camera_angle * numpy.pi / 180)

                return size_x, size_y

            upper_left_x, upper_left_y, lower_right_x, lower_right_y = calibration["reference_marker"]
            reference_marker_height = calibration["reference_marker_height"]
            vertical_camera_angle = calibration["angle_vertical"]

            reference_marker_width = calibration["reference_marker_width"]
            horizontal_camera_angle = calibration["angle_horizontal"]

            # Derived properties
            origin_x, origin_y = _calculate_center()
            pixel_size_x, pixel_size_y = _calculate_pixel_size()  # pixel size in nanometer

            x_axis = numpy.linspace(0, width - 1, width, dtype='f')
            x_axis -= origin_x
            x_axis *= (-pixel_size_x)  # we need the minus to invert the axis

            y_axis = numpy.linspace(0, height - 1, height, dtype='f')
            y_axis -= origin_y
            y_axis *= (-pixel_size_y)  # we need the minus to invert the axis

        self.camera_config.parameters["background_data"] = None
        background_filename = self.camera_config.parameters["image_background"]
        if background_filename:
            background_array = numpy.load(background_filename)
            if background_array is not None:
                if ((background_array.shape[1] != x_axis.shape[0]) or (background_array.shape[0] != y_axis.shape[0])):
                    _logger.info("Invalid background shape for camera %s: %s" % (self.camera_config.get_source(background_array.shape), str()))
                else:
                    self.camera_config.parameters["background_data"] = background_array.astype("uint16", copy=False)

        roi = self.camera_config.parameters["roi"]
        if roi is not None:
            offset_x, size_x, offset_y, size_y = roi

            x_axis =  x_axis[offset_x : offset_x + size_x]
            y_axis = y_axis[offset_y: offset_y + size_y]

            background = self.camera_config.parameters.get("background_data")
            if background is not None:
                self.camera_config.parameters["background_data"] = \
                    background [offset_y:offset_y + size_y, offset_x:offset_x + size_x]


        self.backgroung_image = None

        return x_axis, y_axis
