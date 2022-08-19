import os
from logging import getLogger

import numpy
from bsread.sender import Sender, PUB

from cam_server import config
from cam_server.camera.sender import *
from cam_server.camera.source.common import transform_image
from cam_server.ipc import IpcSender
from cam_server.utils import update_statistics, on_message_sent, init_statistics

_logger = getLogger(__name__)


def get_ipc_address(name):
    if not os.path.exists(config.IPC_FEEDS_FOLDER):
        os.makedirs(config.IPC_FEEDS_FOLDER)
    return "ipc://" + config.IPC_FEEDS_FOLDER + "/cam_server_icp_%s" % (name)


class Camera:

    def __init__(self, camera_config):
        """
        Create EPICS camera source.
        :param camera_config: Config of the camera.
        """
        self.camera_config = camera_config

        # Width and height of the raw image
        self.width_raw = None
        self.height_raw = None
        self.dtype = None

        self.simulate_pulse_id = self.camera_config.get_configuration().get("simulate_pulse_id", False)
        self.check_data = self.camera_config.get_configuration().get("check_data", False)
        self.last_pid = 0

    def get_raw_geometry(self):
        return self.width_raw, self.height_raw

    def get_image(self, raw=False):
        return

    def get_geometry(self):
        width, height = self.get_raw_geometry()
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

    def get_parameters(self):
        return self.camera_config.parameters

    def get_client_timeout(self):
        client_timeout = self.camera_config.get_configuration().get("no_client_timeout")
        if client_timeout is not None:
            return client_timeout
        return config.MFLOW_NO_CLIENTS_TIMEOUT

    def get_connections(self):
        connections = self.camera_config.get_configuration().get("connections")
        try:
            if connections is not None:
                return max(int(connections), 1)
        except:
            _logger.warning("Invalid number of connections (using 1) [%s]" % (self.get_name(),))
        return 1

    def get_buffer_size(self):
        buffer_size = self.camera_config.get_configuration().get("buffer_size")
        try:
            if buffer_size is not None:
                return max(int(buffer_size), 0)
        except:
            _logger.warning("Invalid buffer size (using default: " + str(0) + ") [%s]" % (self.get_name(),))
        return 0

    def get_buffer_threshold(self):
        buffer_threshold = self.camera_config.get_configuration().get("buffer_threshold")
        try:
            if buffer_threshold is not None:
                return min(max(float(buffer_threshold), 0.0), 1.0)
        except:
            _logger.warning("Invalid buffer threshold (using 0.5) [%s]" % (self.get_name(),))
        return 0.5

    def get_dtype(self):
        dtype = self.camera_config.get_configuration().get("dtype")
        if dtype is None:
            if self.dtype  is None:
                return  "uint16"
            return self.dtype
        return dtype

    def get_debug(self):
        debug = self.camera_config.get_configuration().get("debug")
        try:
            return str(debug).lower() == "true"
        except:
            pass
        return False

    def get_queue_size(self):
        queue_size = self.camera_config.get_configuration().get("queue_size")
        try:
            if queue_size is not None:
                return max(int(queue_size), 1)
        except:
            _logger.warning("Invalid number of queue_size (using %d) [%s]" % (config.CAMERA_DEFAULT_QUEUE_SIZE, self.get_name(),))
        return config.CAMERA_DEFAULT_QUEUE_SIZE

    def no_client_timeout(self):
        client_timeout = self.get_client_timeout()
        if client_timeout > 0:
            _logger.info("No client connected to the stream for %d seconds. Closing instance [%s]." %
                         (client_timeout, self.get_name()))
            self.stop_event.set()

    def create_sender(self, stop_event, port):
        self.stop_event = stop_event
        if self.camera_config.get_configuration().get("protocol", "tcp") == "ipc":
            sender = IpcSender(address=get_ipc_address(self.get_name()), mode=PUB, start_pulse_id=self.get_start_pulse_id(), queue_size=self.get_queue_size(),
                             data_header_compression=config.CAMERA_BSREAD_DATA_HEADER_COMPRESSION)
        else:
            sender = Sender(queue_size=self.get_queue_size(), port=port, mode=PUB, start_pulse_id=self.get_start_pulse_id(), data_header_compression=config.CAMERA_BSREAD_DATA_HEADER_COMPRESSION)
        sender.open(no_client_action=self.no_client_timeout, no_client_timeout=self.get_client_timeout())
        return sender

    def abort_on_error(self):
        return self.camera_config.get_configuration().get("abort_on_error", config.ABORT_ON_ERROR)


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
                try:
                    size_y = reference_marker_height / (lower_right_y - upper_left_y)
                    size_y *= numpy.cos(vertical_camera_angle * numpy.pi / 180)

                    size_x = reference_marker_width / (lower_right_x - upper_left_x)
                    size_x *= numpy.cos(horizontal_camera_angle * numpy.pi / 180)
                except:
                    _logger.error("Invalid calibration for camera %s" % (self.camera_config.get_source()))
                    # Dont't abort pipeline if calibration is invalid (division by zero)
                    size_x, size_y = 1.0, 1.0

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
                    _logger.info("Invalid background shape for camera %s: %s" % (self.camera_config.get_source(), str(background_array.shape)))
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


        return x_axis, y_axis

    def get_client_timeout(self):
        client_timeout = self.camera_config.get_configuration().get("no_client_timeout")
        if client_timeout is not None:
            return client_timeout
        return config.MFLOW_NO_CLIENTS_TIMEOUT

    def get_frame_rate(self):
        if "frame_rate" in self.camera_config.get_configuration():
            frame_rate = self.camera_config.get_configuration()["frame_rate"]
            if frame_rate > 0:
                return frame_rate
        return None

    def get_image(self, raw=False):
        value = self.read()
        # Return raw image without any corrections
        if raw:
            return value
        return transform_image(value, self.camera_config)

    def get_pulse_id(self):
        if self.simulate_pulse_id:
            ret = int(time.time() * 100)
            if ret <= self.last_pid:
                ret = self.last_pid+1
            self.last_pid = ret
            return ret
        return None

    def get_start_pulse_id(self):
        return 0
        # return int(time.time() * 100)

    def get_data(self):
        image = self.get_image()
        timestamp = time.time()
        return image, timestamp, self.get_pulse_id()

    def register_channels(self):
        # Register the bsread channels - compress only the image.
        dtype = self.get_dtype()
        x_size, y_size = self.get_geometry()
        self.sender.add_channel("width", metadata={"compression": config.CAMERA_BSREAD_SCALAR_COMPRESSION, "type": "int64"})
        self.sender.add_channel("height", metadata={"compression": config.CAMERA_BSREAD_SCALAR_COMPRESSION, "type": "int64"})
        self.sender.add_channel("timestamp",
                           metadata={"compression": config.CAMERA_BSREAD_SCALAR_COMPRESSION, "type": "float64"})
        self.sender.add_channel("image",
                           metadata={"compression": config.CAMERA_BSREAD_IMAGE_COMPRESSION, "shape": [x_size, y_size],
                                     "type": dtype})
        self.sender.add_channel("x_axis",
                           metadata={"compression": config.CAMERA_BSREAD_SCALAR_COMPRESSION, "shape": [x_size],
                                     "type": "float32"})
        self.sender.add_channel("y_axis",
                           metadata={"compression": config.CAMERA_BSREAD_SCALAR_COMPRESSION, "shape": [y_size],
                                     "type": "float32"})

    def get_send_channels(self, default_channels):
        return default_channels

    ####################################################################################################################
    # VIRTUALS
    ####################################################################################################################
    def verify_camera_online(self):
        return

    def connect(self):
        return

    def disconnect(self):
        return

    def read(self):
        return


####################################################################################################################
# PROCESSING FUNCTION
####################################################################################################################

    def process(self, stop_event, statistics, parameter_queue, port):
        sender = None
        dtype = None
        try:
            init_statistics(statistics)
            self.sender = self.create_sender(stop_event, port)
            self.connect()
            camera_name = self.get_name()
            x_size, y_size = self.get_geometry()
            x_axis, y_axis = self.get_x_y_axis()
            frame_rate = self.get_frame_rate()
            sample_interval = (1.0 / frame_rate) if frame_rate else None

            if not self.check_data:
                self.register_channels()


            # This signals that the camera has suc cessfully started.
            stop_event.clear()

            while not stop_event.is_set():
                if sample_interval:
                    start = time.time()
                image, timestamp, pulse_id = self.get_data()
                frame_size = ((image.size * image.itemsize) if (image is not None) else 0)
                frame_shape = str(x_size) + "x" + str(y_size) + "x" + str(image.itemsize)
                update_statistics(sender, -frame_size, 1 if (image is not None) else 0, frame_shape)

                # In case of receiving error or timeout, the returned data is None.
                if image is None:
                    continue

                change = False
                x,y = self.get_geometry()
                if x!=x_size or y!=y_size:
                    x_size, y_size = self.get_geometry()
                    x_axis, y_axis = self.get_x_y_axis()
                    change = True
                if (dtype is not None) and dtype!=image.dtype:
                    change = True
                dtype = image.dtype
                if change and not self.check_data:
                    self.register_channels()


                default_channels = {"image": image,
                        "timestamp": timestamp,
                        "width": x_size,
                        "height": y_size,
                        "x_axis": x_axis,
                        "y_axis": y_axis}
                data = self.get_send_channels(default_channels)

                try:
                    self.sender.send(data=data, pulse_id=pulse_id, timestamp=timestamp, check_data=self.check_data)
                    on_message_sent()
                except Again:
                    _logger.warning(
                        "Send timeout. Lost image with timestamp '%s' [%s]." % (str(timestamp), camera_name))
                if sample_interval:
                    sleep = sample_interval - (time.time()-start)
                    if (sleep>0):
                        time.sleep(sleep)

        except Exception as e:
            _logger.exception("Error while processing camera stream: %s" % (str(e),))

        finally:
            _logger.info("Stopping transceiver.")

            # Wait for termination / update configuration / etc.
            stop_event.wait()

            try:
                self.disconnect()
            except:
                pass

            if self.sender:
                try:
                    self.sender.close()
                except:
                    pass
