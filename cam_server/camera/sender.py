from logging import getLogger

from bsread.sender import Sender, PUB
from zmq import Again

from cam_server import config

_logger = getLogger(__name__)


def process_epics_camera(stop_event, statistics, parameter_queue,
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

        def process_parameters():
            nonlocal x_size, y_size, x_axis, y_axis
            x_size, y_size = camera.get_geometry()
            x_axis, y_axis = camera.get_x_y_axis()
            sender.add_channel("image", metadata={"compression": config.CAMERA_BSREAD_IMAGE_COMPRESSION,
                                                  "shape": [x_size, y_size],
                                                  "type": "float32"})

        x_size = y_size = x_axis = y_axis = None
        camera.connect()

        sender = Sender(port=port, mode=PUB,
                        data_header_compression=config.CAMERA_BSREAD_DATA_HEADER_COMPRESSION)

        # Register the bsread channels - compress only the image.
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

        process_parameters()
        statistics.counter = 0

        def collect_and_send(image, timestamp):
            nonlocal x_size, y_size, x_axis, y_axis

            # Data to be sent over the stream.
            data = {"image": image,
                    "timestamp": timestamp,
                    "width": x_size,
                    "height": y_size,
                    "x_axis": x_axis,
                    "y_axis": y_axis}

            try:
                sender.send(data=data, timestamp=timestamp, check_data=False)
            except Again:
                _logger.warning("Send timeout. Lost image with timestamp '%s'." % timestamp)

            while not parameter_queue.empty():
                new_parameters = parameter_queue.get()
                camera.camera_config.set_configuration(new_parameters)

                process_parameters()

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


def process_bsread_camera(stop_event, statistics, parameter_queue,
                          camera, port):
    """
    Start the camera stream and receive the incoming bsread streams. This function blocks until stop_event is set.
    :param stop_event: Event when to stop the process.
    :param statistics: Statistics namespace.
    :param parameter_queue: Parameters queue to be passed to the pipeline.
    :param camera: Camera instance to get the stream from.
    :param port: Port to use to bind the output stream.
    """
    sender = None

    try:

        # If there is no client for some time, disconnect.
        def no_client_timeout():
            _logger.info("No client connected to the '%s' stream for %d seconds. Closing instance." %
                         (camera.get_name(), config.MFLOW_NO_CLIENTS_TIMEOUT))
            stop_event.set()

        def process_parameters():
            nonlocal x_size, y_size, x_axis, y_axis
            x_axis, y_axis = camera.get_x_y_axis()
            x_size, y_size = camera.get_geometry()

        # TODO: Use to register proper channels. But be aware that the size and dtype can change during the running.
        # def register_image_channel(size_x, size_y, dtype):
        #     sender.add_channel("image", metadata={"compression": config.CAMERA_BSREAD_IMAGE_COMPRESSION,
        #                                           "shape": [size_x, size_y],
        #                                           "type": dtype})

        x_size = y_size = x_axis = y_axis = None

        sender = Sender(port=port, mode=PUB,
                        data_header_compression=config.CAMERA_BSREAD_DATA_HEADER_COMPRESSION)

        sender.open(no_client_action=no_client_timeout, no_client_timeout=config.MFLOW_NO_CLIENTS_TIMEOUT)

        camera_name = camera.get_name()
        camera_stream = camera.get_stream()

        _logger.info("Connecting to camera '%s' over bsread.", camera_name)

        process_parameters()
        # register_image_channel(x_size, y_size, dtype)

        statistics.counter = 0
        camera_stream.connect()

        # This signals that the camera has successfully started.
        stop_event.clear()

        while not stop_event.is_set():
            try:

                data = camera_stream.receive()

                # In case of receiving error or timeout, the returned data is None.
                if data is None:
                    continue

                image = data.data.data[camera_name + config.EPICS_PV_SUFFIX_IMAGE].value

                # Numpy is slowest dimension first, but bsread is fastest dimension first.
                height, width = image.shape

                pulse_id = data.data.pulse_id

                timestamp_s = data.data.global_timestamp
                timestamp_ns = data.data.global_timestamp_offset
                timestamp = timestamp_s + (timestamp_ns / 1e9)

                data = {
                    "image": image,
                    "height": height,
                    "width": width,
                    "x_axis": x_axis,
                    "y_axis": y_axis,
                    "timestamp": timestamp
                }

                sender.send(data=data, pulse_id=pulse_id, timestamp=timestamp, check_data=True)

                while not parameter_queue.empty():
                    new_parameters = parameter_queue.get()
                    camera.camera_config.set_configuration(new_parameters)

                    process_parameters()

            except:
                _logger.exception("Could not process message.", e)
                stop_event.set()

        _logger.info("Stopping transceiver.")

    except:
        _logger.exception("Error while processing camera stream.")

    finally:
        # Wait for termination / update configuration / etc.
        stop_event.wait()

        if camera_stream:
            camera.disconnect()

        if sender:
            sender.close()


source_type_to_sender_function_mapping = {
    "epics": process_epics_camera,
    "simulation": process_epics_camera,
    "bsread": process_bsread_camera
}


def get_sender_function(source_type_name):
    if source_type_name not in source_type_to_sender_function_mapping:
        raise ValueError("source_type '%s' not present in sender function pv_to_stream_mapping. Available: %s." %
                         (source_type_name, list(source_type_to_sender_function_mapping.keys())))

    return source_type_to_sender_function_mapping[source_type_name]
