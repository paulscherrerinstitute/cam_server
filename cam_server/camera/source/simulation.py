import time
from threading import Event, Thread

import numpy

from logging import getLogger

from cam_server import config
from cam_server.camera.source.epics import CameraEpics

_logger = getLogger(__name__)


class CameraSimulation(CameraEpics):
    """
    Camera simulation for debugging purposes.
    """

    def verify_camera_online(self):
        return True

    # TODO: Make this a config.
    def __init__(self, camera_config, size_x=1280, size_y=960, number_of_dead_pixels=100, noise=0.1,
                 beam_size_x=100, beam_size_y=20, frame_rate=10, dtype="int16"):
        """
        Initialize the camera simulation.
        :param size_x: Image width.
        :param size_y: Image height.
        :param number_of_dead_pixels: Number of simulated dead pixels.
        :param noise: How much noise to introduce.
        :param beam_size_x: Beam width.
        :param beam_size_y: Beam height.
        :param frame_rate: How many frames, in mhz, does the simulation outputs.
        """
        super(CameraSimulation, self).__init__(camera_config)

        if "frame_rate" in camera_config.get_configuration():
            frame_rate = camera_config.get_configuration()["frame_rate"]
        if "size_x" in camera_config.get_configuration():
            size_x = camera_config.get_configuration()["size_x"]
        if "size_y" in camera_config.get_configuration():
            size_y = camera_config.get_configuration()["size_y"]
        if "dtype" in camera_config.get_configuration():
            dtype = camera_config.get_configuration()["dtype"]

        self.frame_rate = frame_rate
        self.dtype = dtype
        self.width_raw = size_x
        self.height_raw = size_y
        self.noise = noise  # double {0,1} noise amplification factor
        self.dead_pixels = self._generate_dead_pixels(number_of_dead_pixels)
        self.beam_size_x = beam_size_x
        self.beam_size_y = beam_size_y

        # Functions to call in simulation.
        self.callback_functions = []
        self.simulation_thread = None
        self.simulation_stop_event = Event()
        self.image_type =  camera_config.get_configuration().get("image_type")
        self.raw = self.image_type in ["raw", "static_raw"]
        self.static = self.image_type in ["static_beam", "static_raw"]


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
            image = numpy.zeros((self.height_raw, self.width_raw))
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

        image = image.astype(self.dtype)

        return self._get_image(image, raw=raw)

    def connect(self):
        # Thread already exists.
        if self.simulation_thread:
            return

        self.simulation_stop_event.clear()

        def call_callbacks(stop_event):
            image = None
            next_img_timestamp=time.time()
            interval = 1.0/self.frame_rate
            while not stop_event.is_set():
                try:
                    # Same timestamp as used by PyEpics.
                    timestamp = time.time()
                    if timestamp>=next_img_timestamp:
                        if (image is None) or (not self.static):
                            image = self.get_image(self.raw)

                        for callback in self.callback_functions:
                             callback(image, timestamp)
                        #Try keep time base in long
                        next_img_timestamp = next_img_timestamp + interval
                        #Does not try to compensate lagging
                        #next_img_timestamp = timestamp + interval

                    time.sleep(interval / 10)

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
