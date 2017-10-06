import epics
import numpy

from logging import getLogger

from cam_server.camera.source.utils import get_calibrated_axis

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

    def verify_camera_online(self):
        camera_prefix = self.camera_config.parameters["source"]
        camera_init_pv = camera_prefix + ":INIT"

        channel_init = epics.PV(camera_init_pv)
        channel_init_value = channel_init.get(as_string=True)
        channel_init.disconnect()

        if channel_init_value != 'INIT':
            raise RuntimeError("Camera with prefix {} not online - Status {}".format(camera_prefix, channel_init_value))

    def connect(self):

        self.verify_camera_online()

        # Retrieve with and height of cam_server image.
        camera_width_pv = self.camera_config.parameters["source"] + ":WIDTH"
        camera_height_pv = self.camera_config.parameters["source"] + ":HEIGHT"

        _logger.debug("Checking camera WIDTH '%s' and HEIGHT '%s' PV.", camera_width_pv, camera_height_pv)

        channel_width = epics.PV(camera_width_pv)
        channel_height = epics.PV(camera_height_pv)

        self.width_raw = int(channel_width.get(timeout=4))
        self.height_raw = int(channel_height.get(timeout=4))

        if not self.width_raw or not self.height_raw:
            raise RuntimeError("Could not fetch width and height for cam_server:{}".format(
                self.camera_config.parameters["source"]))

        channel_width.disconnect()
        channel_height.disconnect()

        # Connect image channel
        self.channel_image = epics.PV(self.camera_config.parameters["source"] + ":FPICTURE", auto_monitor=True)
        self.channel_image.wait_for_connection(1.0)  # 1 second default connection timeout

        if not self.channel_image.connected:
            raise RuntimeError("Could not connect to: {}".format(self.channel_image.pvname))

    def disconnect(self):
        self.clear_callbacks()

        if self.channel_image:
            self.channel_image.disconnect()

        self.channel_image = None

    def add_callback(self, callback_function):

        def _callback(value, timestamp, status, **kwargs):
            callback_function(self._get_image(value), timestamp)

        self.channel_image.add_callback(_callback)

    def _get_image(self, value, raw=False):

        if value is None:
            return None

        # Convert type - we are using f because of better performance
        # floats (32bit-ones) are way faster to calculate than 16 bit ints, actually even faster than native
        # int type (32/64uint) since we can leverage SIMD instructions (SSE/SSE2 on Intels).
        value = value.astype('u2').astype(numpy.float32)

        # Shape image
        value = value[:(self.width_raw * self.height_raw)].reshape((self.height_raw, self.width_raw))

        # Return raw image without any corrections
        if raw:
            return value

        # Correct image
        if self.camera_config.parameters["mirror_x"]:
            value = numpy.fliplr(value)

        if self.camera_config.parameters["mirror_y"]:
            value = numpy.flipud(value)

        value = numpy.rot90(value, self.camera_config.parameters["rotate"])

        return numpy.ascontiguousarray(value)

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
        # We cannot get the geometry until we connect to the camera for the first time.
        if self.width_raw is None or self.height_raw is None:
            self.connect()
            self.disconnect()

        rotate = self.camera_config.parameters["rotate"]
        if rotate == 1 or rotate == 3:
            # If rotating by 90 degree, height becomes width.
            return self.height_raw, self.width_raw
        else:
            return self.width_raw, self.height_raw

    def get_name(self):
        return self.camera_config.get_name()

    def clear_callbacks(self):
        if self.channel_image:
            self.channel_image.clear_callbacks()

    def get_x_y_axis(self):
        calibration = self.camera_config.parameters["camera_calibration"]
        width, height = self.get_geometry()

        return get_calibrated_axis(calibration, height, width)
