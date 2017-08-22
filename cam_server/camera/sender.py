from logging import getLogger

import numpy
from bsread.sender import Sender, PUB
from zmq import Again

from cam_server import config

_logger = getLogger(__name__)


def process_camera_stream(stop_event, statistics, parameter_queue,
                          camera, port):
    """
    Start the camera stream and listen for image monitors. This function blocks until stop_event is set.
    :param stop_event: Event when to stop the process.
    :param statistics: Statistics namespace.
    :param parameter_queue: Parameters queue to be passed to the pipeline.
    :param camera: Camera instance to get the images from.
    :param port: Port to use to bind the output stream.
    """
    sender = None

    try:

        # If there is no client for some time, disconnect.
        def no_client_timeout():
            _logger.info("No client connected to the '%s' stream for %d seconds. Closing instance." %
                         (camera.get_name(), config.MFLOW_NO_CLIENTS_TIMEOUT))
            stop_event.set()

        camera.connect()
        x_size, y_size = camera.get_geometry()
        x_axis, y_axis = camera.get_x_y_axis()

        sender = Sender(port=port, mode=PUB,
                        data_header_compression=config.CAMERA_BSREAD_DATA_HEADER_COMPRESSION)

        # Register the bsread channels - compress only the image.
        sender.add_channel("image", metadata={"compression": config.CAMERA_BSREAD_IMAGE_COMPRESSION,
                                              "shape": [y_size, x_size],
                                              "type": "float32"})

        sender.add_channel("width", metadata={"compression": config.CAMERA_BSREAD_SCALAR_COMPRESSION,
                                              "type": "int64"})

        sender.add_channel("height", metadata={"compression": config.CAMERA_BSREAD_SCALAR_COMPRESSION,
                                               "type": "int64"})

        sender.add_channel("timestamp", metadata={"compression": config.CAMERA_BSREAD_SCALAR_COMPRESSION,
                                                  "type": "float64"})

        sender.add_channel("x_axis", metadata={"compression": config.CAMERA_BSREAD_SCALAR_COMPRESSION,
                                               "type": "float32"})

        sender.add_channel("y_axis", metadata={"compression": config.CAMERA_BSREAD_SCALAR_COMPRESSION,
                                               "type": "float32"})

        sender.open(no_client_action=no_client_timeout, no_client_timeout=config.MFLOW_NO_CLIENTS_TIMEOUT)

        statistics.counter = 0

        def collect_and_send(image, timestamp):

            while not parameter_queue.empty():
                new_parameters = parameter_queue.get()
                camera.camera_config.set_configuration(new_parameters)

                nonlocal x_size, y_size, x_axis, y_axis
                x_size, y_size = camera.get_geometry()
                x_axis, y_axis = camera.get_x_y_axis()

            # Data to be sent over the stream.
            data = {"image": image,
                    "timestamp": timestamp,
                    "width": x_size,
                    "height": y_size,
                    "x_axis": x_axis,
                    "y_axis": y_axis}

            try:
                sender.send(data=data, check_data=False)
            except Again:
                _logger.warning("Send timeout. Lost image with timestamp '%s'." % timestamp)

        camera.add_callback(collect_and_send)

        # This signals that the camera has successfully started.
        stop_event.clear()

    except:
        _logger.exception("Error while processing camera stream.")

    finally:

        # Wait for termination / update configuration / etc.
        stop_event.wait()

        camera.disconnect()

        if sender:
            sender.close()
