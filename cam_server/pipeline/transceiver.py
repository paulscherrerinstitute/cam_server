from logging import getLogger

from bsread import Source, PUB, SUB, PUSH
from bsread.sender import Sender

from cam_server import config
from cam_server.pipeline.data_processing.processor import process_image
from cam_server.utils import get_host_port_from_stream_address

_logger = getLogger(__name__)


def processing_pipeline(stop_event, statistics, parameter_queue,
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

            background_array = background_manager.get_background(background_id)

        size_x, size_y = cam_client.get_camera_geometry(pipeline_config.get_camera_name())

        image_region_of_interest = parameters.get("image_region_of_interest")
        if image_region_of_interest:
            _, size_x, _, size_y = image_region_of_interest

        _logger.debug("Image width %d and height %d.", size_x, size_y)

        return parameters, background_array

    source = None
    sender = None

    try:
        pipeline_parameters, image_background_array = process_pipeline_parameters()

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
                    pipeline_parameters, image_background_array = process_pipeline_parameters()

                data = source.receive()

                # In case of receiving error or timeout, the returned data is None.
                if data is None:
                    continue

                image = data.data.data["image"].value
                timestamp = data.data.data["timestamp"].value
                x_axis = data.data.data["x_axis"].value
                y_axis = data.data.data["y_axis"].value

                processed_data = process_image(image, timestamp, x_axis, y_axis,
                                               pipeline_parameters, image_background_array)

                processed_data["width"] = processed_data["image"].shape[1]
                processed_data["height"] = processed_data["image"].shape[0]

                sender.send(data=processed_data)

            except:
                _logger.exception("Could not process message.")
                stop_event.set()

        _logger.info("Stopping transceiver.")

    except:
        _logger.exception("Exception while trying to start the receive and process thread.")
        raise

    finally:
        if source:
            source.disconnect()

        if sender:
            sender.close()


def store_pipeline(stop_event, statistics, parameter_queue,
                   cam_client, pipeline_config, output_stream_port, background_manager):
    def no_client_timeout():
        _logger.warning("No client connected to the pipeline stream for %d seconds. Closing instance." %
                        config.MFLOW_NO_CLIENTS_TIMEOUT)
        stop_event.set()

    source = None
    sender = None

    try:

        camera_stream_address = cam_client.get_camera_stream(pipeline_config.get_camera_name())
        _logger.debug("Connecting to camera stream address %s.", camera_stream_address)

        source_host, source_port = get_host_port_from_stream_address(camera_stream_address)

        source = Source(host=source_host, port=source_port, receive_timeout=config.PIPELINE_RECEIVE_TIMEOUT, mode=SUB)

        source.connect()

        _logger.debug("Opening output stream on port %d.", output_stream_port)

        sender = Sender(port=output_stream_port, mode=PUSH,
                        data_header_compression=config.CAMERA_BSREAD_DATA_HEADER_COMPRESSION, block=False)

        sender.open(no_client_action=no_client_timeout, no_client_timeout=config.MFLOW_NO_CLIENTS_TIMEOUT)
        # TODO: Register proper channels.

        # Indicate that the startup was successful.
        stop_event.clear()

        _logger.debug("Transceiver started.")

        while not stop_event.is_set():
            try:

                data = source.receive()

                # In case of receiving error or timeout, the returned data is None.
                if data is None:
                    continue

                forward_data = {"image": data.data.data["image"].value,
                                "timestamp": data.data.data["timestamp"].value,
                                "x_axis": data.data.data["x_axis"].value,
                                "y_axis": data.data.data["y_axis"].value,
                                "width": data.data.data["width"].value,
                                "height": data.data.data["height"].value}

                sender.send(data=forward_data)

            except:
                _logger.exception("Could not process message.")
                stop_event.set()

        _logger.info("Stopping transceiver.")

    except:
        _logger.exception("Exception while trying to start the receive and process thread.")
        raise

    finally:
        if source:
            source.disconnect()

        if sender:
            sender.close()


pipeline_name_to_pipeline_function_mapping = {
    "processing": processing_pipeline,
    "store": store_pipeline
}


def get_pipeline_function(pipeline_type_name):
    if pipeline_type_name not in pipeline_name_to_pipeline_function_mapping:
        raise ValueError("pipeline_type '%s' not present in mapping. Available: %s." %
                         (pipeline_type_name, list(pipeline_name_to_pipeline_function_mapping.keys())))

    return pipeline_name_to_pipeline_function_mapping[pipeline_type_name]
