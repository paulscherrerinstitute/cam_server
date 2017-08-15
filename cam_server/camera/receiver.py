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

        # Width and height of the corrected image
        self.width = 0
        self.height = 0
        # Width and height of the raw image
        self.width_raw = 0
        self.height_raw = 0

        self.channel_image = None

    def connect(self):

        # Check cam_server status
        channel_init = epics.PV(self.camera_config.camera_config.parameters["prefix"] + ":INIT")
        if channel_init.get(as_string=True) != 'INIT':
            raise RuntimeError("Camera {} not online - Status {}".format(self.prefix, channel_init.get(as_string=True)))

        channel_init.disconnect()

        # Retrieve with and height of cam_server image
        channel_width = epics.PV(self.camera_config.parameters["prefix"] + ":WIDTH")
        channel_height = epics.PV(self.camera_config.parameters["prefix"] + ":HEIGHT")

        self.width_raw = int(channel_width.get(timeout=4))
        self.height_raw = int(channel_height.get(timeout=4))

        if not self.width_raw or not self.height_raw:
            raise RuntimeError("Could not fetch width and height for cam_server:{}".format(self.prefix))

        channel_width.disconnect()
        channel_height.disconnect()

        # Connect image channel
        self.channel_image = epics.PV(self.camera_config.parameters["prefix"] + ":FPICTURE", auto_monitor=True)
        self.channel_image.wait_for_connection(1.0)  # 1 second default connection timeout

        if not self.channel_image.connected:
            raise RuntimeError("Could not connect to: {}".format(self.channel_image.pvname))

        # Set correct width and height of the corrected image
        if self.rotate == 1 or self.rotate == 3:
            self.width = self.height_raw
            self.height = self.width_raw
        else:
            self.width = self.width_raw
            self.height = self.height_raw

    def disconnect(self):
        self.clear_callbacks()
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
        return self.width, self.height

    def get_name(self):
        return self.camera_config.get_name()

    def clear_callbacks(self):
        self.channel_image.clear_callbacks()

    def get_x_y_axis(self):

        if not self.width or not self.height:
            raise RuntimeError('Width and height of the cam_server not known yet - connect first')

        x_axis = numpy.linspace(0, self.width - 1, self.width, dtype='f8')
        y_axis = numpy.linspace(0, self.height - 1, self.height, dtype='f8')

        return x_axis, y_axis


class CameraSimulation:
    """
    Camera simulation for debugging purposes.
    """
    def __init__(self, size_x=1280, size_y=960, number_of_dead_pixels=100, noise=0.1,
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

        self.frame_rate = frame_rate
        self.size_x = size_x
        self.size_y = size_y
        self.noise = noise  # double {0,1} noise amplification factor
        self.dead_pixels = self.generate_dead_pixels(number_of_dead_pixels)
        self.beam_size_x = beam_size_x
        self.beam_size_y = beam_size_y

        # Functions to call in simulation.
        self.callback_functions = []
        self.simulation_interval = simulation_interval
        self.simulation_thread = None
        self.simulation_stop_event = Event()

    def generate_dead_pixels(self, number_of_dead_pixel):
        dead_pixels = numpy.zeros((self.size_y, self.size_x))

        for _ in range(number_of_dead_pixel):
            x = numpy.random.randint(0, self.size_y)
            y = numpy.random.randint(0, self.size_x)
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
                                    self.size_y)
            beam_y = numpy.linspace(-self.beam_size_y + numpy.random.rand(),
                                    self.beam_size_y + numpy.random.rand(),
                                    self.size_x)
            x, y = numpy.meshgrid(beam_y, beam_x)
            image = numpy.exp(-(x ** 2 + y ** 2))

        # Add some noise
        if self.noise:
            image += numpy.random.random((self.size_y, self.size_x)) * self.noise

        # Add dead pixels
        image += self.dead_pixels

        image.clip(0, 0.9, out=image)
        image *= (numpy.power(2, 16) - 1)

        # Convert to float (for better performance)
        image = image.astype('f')
        return image

    def get_x_y_axis(self):
        x_axis = numpy.linspace(0, self.size_y - 1, self.size_y, dtype='f8')
        y_axis = numpy.linspace(0, self.size_x - 1, self.size_x, dtype='f8')
        return y_axis, x_axis

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

    def get_geometry(self):
        return self.size_x, self.size_y

    def get_name(self):
        return "simulation"

    def add_callback(self, callback_function):
        self.callback_functions.append(callback_function)

    def clear_callbacks(self):
        self.callback_functions.clear()
