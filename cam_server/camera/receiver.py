import epics
import numpy


class Camera:

    def __init__(self, prefix, mirror_x=False, mirror_y=False, rotate=0):
        """
        Parameters
        ----------
        prefix      prefix of cam_server channel
        mirror_x    mirror image on x axis
        mirror_y    mirror image on y axis
        rotate      number of 90deg rotation 0=0deg 1=90deg, 2=180deg, ...
        """

        self.prefix = prefix
        self.mirror_x = mirror_x
        self.mirror_y = mirror_y
        self.rotate = rotate

        # Width and height of the corrected image
        self.width = 0
        self.height = 0
        # Width and height of the raw image
        self.width_raw = 0
        self.height_raw = 0

        # Offsets are given by the ROI set on the cam_server
        self.reference_offset_x = 0
        self.reference_offset_y = 0

        self.channel_image = None

    def connect(self):

        # Check cam_server status
        channel_init = epics.PV(self.prefix + ":INIT")
        if channel_init.get(as_string=True) != 'INIT':
            raise RuntimeError("Camera {} not online - Status {}".format(self.prefix, channel_init.get(as_string=True)))

        channel_init.disconnect()

        # Retrieve with and height of cam_server image
        channel_width = epics.PV(self.prefix + ":WIDTH")
        channel_height = epics.PV(self.prefix + ":HEIGHT")

        self.width_raw = int(channel_width.get(timeout=4))
        self.height_raw = int(channel_height.get(timeout=4))

        if not self.width_raw or not self.height_raw:
            raise RuntimeError("Could not fetch width and height for cam_server:{}".format(self.prefix))

        channel_width.disconnect()
        channel_height.disconnect()

        # Connect image channel
        self.channel_image = epics.PV(self.prefix + ":FPICTURE", auto_monitor=True)
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
        if self.mirror_x:
            value = numpy.fliplr(value)

        if self.mirror_y:
            value = numpy.flipud(value)

        value = numpy.rot90(value, self.rotate)

        return value

    def get_image(self, raw=False):
        value = self.channel_image.get()
        return self._get_image(value, raw=raw)

    def get_geometry(self):
        return self.width, self.height

    def get_name(self):
        return self.prefix

    def clear_callbacks(self):
        self.channel_image.clear_callbacks()

    def get_x_y_axis(self):

        if not self.width or not self.height:
            raise RuntimeError('Width and height of the cam_server not known yet - connect first')

        x_axis = numpy.linspace(0, self.width - 1, self.width, dtype='f')
        y_axis = numpy.linspace(0, self.height - 1, self.height, dtype='f')

        return x_axis, y_axis


class CameraSimulation:
    """
    Camera simulation for debugging purposes.
    """
    def __init__(self, size_x=1280, size_y=960, number_of_dead_pixels=100, noise=0.1,
                 beam_size_x=100, beam_size_y=20, frame_rate=10):
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

        self.frame_rate = frame_rate
        self.size_x = size_x
        self.size_y = size_y
        self.noise = noise  # double {0,1} noise amplification factor
        self.dead_pixels = self.generate_dead_pixels(number_of_dead_pixels)
        self.beam_size_x = beam_size_x
        self.beam_size_y = beam_size_y

        # Functions to call in simulation.
        self.callback_functions = []
        self.simulation_thread = None

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
        x_axis = numpy.linspace(0, self.size_y - 1, self.size_y, dtype='f')
        y_axis = numpy.linspace(0, self.size_x - 1, self.size_x, dtype='f')
        return y_axis, x_axis

    def connect(self):  # NOOP - Just to match signature of cam_server
        pass

    def disconnect(self):  # NOOP - Just to match signature of cam_server
        pass

    def get_geometry(self):
        return self.size_x, self.size_y

    def get_name(self):
        return "simulation"

    def add_callback(self, callback_function):
        self.callback_functions.append(callback_function)

    def clear_callbacks(self):
        self.callback_functions.clear()


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    source = CameraSimulation()

    images = []
    for i in range(100):
        images.append(source.get_image())

    print('Image generation done')

    # image = source.get_image()
    im = plt.imshow(images[0])

    plt.ion()
    plt.show()

    for image in images:
        image = source.get_image()
        im.set_data(image)
        plt.pause(0.01)
