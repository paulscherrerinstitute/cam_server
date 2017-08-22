from logging import getLogger
from threading import Event, Thread

import epics
import numpy
import time

_logger = getLogger(__name__)


class Camera:

    def __init__(self, camera_config):
        """
        Parameters
        ----------
        prefix      prefix of cam_server channel
        mirror_x    mirror image on x axis
        mirror_y    mirror image on y axis
        rotate      number of 90deg rotation 0=0deg 1=90deg, 2=180deg, ...
        """

        self.camera_config = camera_config

        # Width and height of the raw image
        self.width_raw = 0
        self.height_raw = 0

        self.channel_image = None

    def verify_camera_online(self):
        camera_prefix = self.camera_config.parameters["prefix"]
        camera_init_pv = camera_prefix + ":INIT"

        channel_init = epics.PV(camera_init_pv)
        channel_init_value = channel_init.get(as_string=True)
        channel_init.disconnect()

        if channel_init_value != 'INIT':
            raise RuntimeError("Camera with prefix {} not online - Status {}".format(camera_prefix, channel_init_value))

    def connect(self):

        self.verify_camera_online()

        # Retrieve with and height of cam_server image.
        camera_width_pv = self.camera_config.parameters["prefix"] + ":WIDTH"
        camera_height_pv = self.camera_config.parameters["prefix"] + ":HEIGHT"

        _logger.debug("Checking camera WIDTH '%s' and HEIGHT '%s' PV.", camera_width_pv, camera_height_pv)

        channel_width = epics.PV(camera_width_pv)
        channel_height = epics.PV(camera_height_pv)

        self.width_raw = int(channel_width.get(timeout=4))
        self.height_raw = int(channel_height.get(timeout=4))

        if not self.width_raw or not self.height_raw:
            raise RuntimeError("Could not fetch width and height for cam_server:{}".format(
                self.camera_config.parameters["prefix"]))

        channel_width.disconnect()
        channel_height.disconnect()

        # Connect image channel
        self.channel_image = epics.PV(self.camera_config.parameters["prefix"] + ":FPICTURE", auto_monitor=True)
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

        return value

    def get_image(self, raw=False):
        value = self.channel_image.get()
        return self._get_image(value, raw=raw)

    def get_geometry(self):
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

            return x_axis, y_axis

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

        return x_axis, y_axis


class CameraSimulation(Camera):
    """
    Camera simulation for debugging purposes.
    """
    def verify_camera_online(self):
        return True

    def __init__(self, camera_config, size_x=1280, size_y=960, number_of_dead_pixels=100, noise=0.1,
                 beam_size_x=100, beam_size_y=20, frame_rate=10, simulation_interval=0.1):
        """
        Initialize the camera simulation.
        :param size_x: Image width.
        :param size_y: Image height.
        :param number_of_dead_pixels: Number of simulated dead pixels.
        :param noise: How much noise to introduce.
        :param beam_size_x: Beam width.
        :param beam_size_y: Beam height.
        :param frame_rate: How many frames, in mhz, does the simulation outputs.
        :param simulation_interval: Interval between frames on the simulated camera.
        """
        super(CameraSimulation, self).__init__(camera_config)

        self.frame_rate = frame_rate
        self.width_raw = size_x
        self.height_raw = size_y
        self.noise = noise  # double {0,1} noise amplification factor
        self.dead_pixels = self._generate_dead_pixels(number_of_dead_pixels)
        self.beam_size_x = beam_size_x
        self.beam_size_y = beam_size_y

        # Functions to call in simulation.
        self.callback_functions = []
        self.simulation_interval = simulation_interval
        self.simulation_thread = None
        self.simulation_stop_event = Event()

    def _generate_dead_pixels(self, number_of_dead_pixel):
        dead_pixels = numpy.zeros((self.height_raw, self.width_raw))

        for _ in range(number_of_dead_pixel):
            x = numpy.random.randint(0, self.height_raw)
            y = numpy.random.randint(0, self.width_raw)
            dead_pixels[x, y] = 1

        return dead_pixels

    def get_image(self, raw=False):
        """
        Get the simulated image.
        :param raw: If true, return a simulated camera wihtout the beam (just noise).
        :return: Camera image.
        """

        if raw:
            image = numpy.zeros((self.size_y, self.size_x))
        else:
            beam_x = numpy.linspace(-self.beam_size_x + numpy.random.rand(),
                                    self.beam_size_x + numpy.random.rand(),
                                    self.height_raw)
            beam_y = numpy.linspace(-self.beam_size_y + numpy.random.rand(),
                                    self.beam_size_y + numpy.random.rand(),
                                    self.width_raw)
            x, y = numpy.meshgrid(beam_y, beam_x)
            image = numpy.exp(-(x ** 2 + y ** 2))

        # Add some noise
        if self.noise:
            image += numpy.random.random((self.height_raw, self.width_raw)) * self.noise

        # Add dead pixels
        image += self.dead_pixels

        image.clip(0, 0.9, out=image)
        image *= (numpy.power(2, 16) - 1)

        return self._get_image(image, raw=raw)

    def connect(self):
        # Thread already exists.
        if self.simulation_thread:
            return

        self.simulation_stop_event.clear()

        def call_callbacks(stop_event):

            while not stop_event.is_set():
                try:
                    image = self.get_image()
                    # Same timestamp as used by PyEpics.
                    timestamp = time.time()

                    for callback in self.callback_functions:
                        callback(image, timestamp)

                    time.sleep(self.simulation_interval)

                except:
                    _logger.exception("Error occurred in camera simulation.")

        self.simulation_thread = Thread(target=call_callbacks, args=(self.simulation_stop_event,))
        self.simulation_thread.start()

    def disconnect(self):
        if not self.simulation_thread:
            return

        self.clear_callbacks()
        self.simulation_stop_event.set()
        self.simulation_thread.join()
        self.simulation_thread = None

    def add_callback(self, callback_function):
        self.callback_functions.append(callback_function)

    def clear_callbacks(self):
        self.callback_functions.clear()
