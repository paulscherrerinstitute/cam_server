from logging import getLogger

import numpy
from bsread import Source, PUB, SUB
from bsread.sender import Sender

from cam_server import config
from cam_server.pipeline.data_processing.processor import process_image
from cam_server.utils import get_host_port_from_stream_address
from zmq import Again

_logger = getLogger(__name__)


def receive_process_send(stop_event, statistics, parameter_queue,
                         cam_client, pipeline_config, output_stream_port, background_manager):
    def no_client_timeout():
        _logger.warning("No client connected to the pipeline stream for %d seconds. Closing instance." %
                        config.MFLOW_NO_CLIENTS_TIMEOUT)
        stop_event.set()

    def process_pipeline_parameters():
        parameters = pipeline_config.get_configuration()
        _logger.debug("Processing pipeline parameters %s.", parameters)

        background_array = None
        if parameters.get("image_background_enable"):
            background_id = pipeline_config.get_background_id()
            _logger.debug("Image background enabled. Using background_id %s.", background_id)

            background_array = background_manager.get_background()

        size_x, size_y = cam_client.get_camera_geometry(pipeline_config.get_camera_name())

        calibration = parameters.get("calibration")
        if calibration:
            axis_x, axis_y = get_calibrated_axis(size_x, size_y, calibration)
        else:
            axis_x = numpy.linspace(0, size_x - 1, size_x, dtype='f')
            axis_y = numpy.linspace(0, size_y - 1, size_y, dtype='f')

        image_region_of_interest = parameters.get("image_region_of_interest")
        if image_region_of_interest:
            _, size_x, _, size_y = image_region_of_interest

        _logger.debug("Image width %d and height %d.", size_x, size_y)

        return parameters, background_array, axis_x, axis_y, size_x, size_y

    try:
        pipeline_parameters, image_background_array, x_axis, y_axis, x_size, y_size = process_pipeline_parameters()

        camera_stream_address = cam_client.get_camera_stream(pipeline_config.get_camera_name())
        _logger.debug("Connecting to camera stream address %s.", camera_stream_address)

        source_host, source_port = get_host_port_from_stream_address(camera_stream_address)

        source = Source(host=source_host, port=source_port, receive_timeout=config.PIPELINE_RECEIVE_TIMEOUT, mode=SUB)
        source.connect()

        _logger.debug("Opening output stream on port %d.", output_stream_port)

        sender = Sender(port=output_stream_port, mode=PUB,
                        data_header_compression=config.CAMERA_BSREAD_DATA_HEADER_COMPRESSION)

        sender.open(no_client_action=no_client_timeout, no_client_timeout=config.MFLOW_NO_CLIENTS_TIMEOUT)
        # TODO: Register proper channels.

        # Indicate that the startup was successful.
        stop_event.clear()

        _logger.debug("Transceiver started.")

        while not stop_event.is_set():
            try:
                while not parameter_queue.empty():
                    new_parameters = parameter_queue.get()
                    pipeline_config.set_configuration(new_parameters)
                    pipeline_parameters, image_background_array, x_axis, y_axis, x_size, y_size \
                        = process_pipeline_parameters()

                try:
                    data = source.receive()
                except Again:
                    _logger.error("Source '%s:%d' did not provide message in time. Aborting.", source_host, source_port)
                    stop_event.set()

                # In case of receiving error, the returned data is None.
                if data is None:
                    continue

                image = data.data.data["image"].value
                timestamp = data.data.data["timestamp"].value

                processed_data = process_image(image, timestamp, x_axis, y_axis,
                                               pipeline_parameters, image_background_array)

                processed_data["width"] = x_size
                processed_data["height"] = y_size

                sender.send(data=processed_data)

            except:
                _logger.exception("Could not process message.")
                stop_event.set()

        _logger.debug("Stopping transceiver.")

        source.disconnect()
        sender.close()

    except:
        _logger.exception("Exception while trying to start the receive and process thread.")
        raise


def get_calibrated_axis(width, height, calibration):
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
