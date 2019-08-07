from logging import getLogger
from importlib import import_module
from imp import load_source
import time
import sys

from bsread import Source, PUB, SUB, PUSH
from bsread.sender import Sender


from cam_server import config
from cam_server.pipeline.data_processing.processor import process_image
from cam_server.utils import get_host_port_from_stream_address, set_statistics, init_statistics
from cam_server.pipeline.data_processing.functions import chunk_copy, rotate, is_number

_logger = getLogger(__name__)


def processing_pipeline(stop_event, statistics, parameter_queue,
                        cam_client, pipeline_config, output_stream_port, background_manager):
    camera_name = pipeline_config.get_camera_name()
    log_tag = " [" + str(camera_name) + " | " + str(pipeline_config.get_name()) + ":" + str(output_stream_port) + "]"
    source = None

    def no_client_timeout():
        _logger.warning("No client connected to the pipeline stream for %d seconds. Closing instance. %s",
                        config.MFLOW_NO_CLIENTS_TIMEOUT, log_tag)
        stop_event.set()

    def connect_to_camera():
        nonlocal source
        camera_stream_address = cam_client.get_instance_stream(pipeline_config.get_camera_name())
        _logger.warning("Connecting to camera stream address %s. %s", camera_stream_address, log_tag)
        source_host, source_port = get_host_port_from_stream_address(camera_stream_address)

        source = Source(host=source_host, port=source_port,
                        receive_timeout=config.PIPELINE_RECEIVE_TIMEOUT, mode=SUB)
        source.connect()




    def process_pipeline_parameters():
        parameters = pipeline_config.get_configuration()

        _logger.debug("Processing pipeline parameters %s. %s", parameters, log_tag)

        background_array = None
        if parameters.get("image_background_enable"):
            background_id = pipeline_config.get_background_id()
            _logger.debug("Image background enabled. Using background_id %s. %s", background_id, log_tag)

            background_array = background_manager.get_background(background_id)

        size_x, size_y = cam_client.get_camera_geometry(pipeline_config.get_camera_name())

        image_region_of_interest = parameters.get("image_region_of_interest")
        if image_region_of_interest:
            _, size_x, _, size_y = image_region_of_interest

        _logger.debug("Image width %d and height %d. %s", size_x, size_y, log_tag)

        if not parameters.get("camera_timeout"):
            parameters["camera_timeout"] = 10.0

        if parameters.get("rotation"):
            if not isinstance(parameters.get("rotation"), dict):
                parameters["rotation"] = {"angle":float(parameters.get("rotation")), "order":1, "mode":"0.0"}
            if not parameters["rotation"].get("angle"):
               parameters["rotation"] = None
            elif not is_number(parameters["rotation"]["angle"]) or (float(parameters["rotation"]["angle"]) == 0):
                parameters["rotation"] = None
            else:
                if not parameters["rotation"].get("order"):
                    parameters["rotation"]["order"] = 1
                if not parameters["rotation"].get("mode"):
                    parameters["rotation"]["mode"] = "0.0"

        return parameters, background_array

    functions = {}

    def get_function(name, reload=False):
        if not name:
            return process_image  # default
        try:
            f = functions.get(name)
            if (not f) or reload:
                if (not f):
                    _logger.info("Importing function %s. %s", name, log_tag)
                else:
                    _logger.debug("Reloading function %s. %s", name, log_tag)
                if '/' in name:
                    mod = load_source('mod', name)
                else:
                    mod = import_module("cam_server.pipeline.data_processing." + str(name))
                functions[name] = f = mod.process_image
            return f
        except:
            _logger.exception("Could not import function: %s. %s", str(name), log_tag)
            return None

    source, sender = None, None
    try:
        init_statistics(statistics)

        pipeline_parameters, image_background_array = process_pipeline_parameters()

        connect_to_camera()

        _logger.debug("Opening output stream on port %d. %s", output_stream_port, log_tag)

        sender = Sender(port=output_stream_port, mode=PUB,
                        data_header_compression=config.CAMERA_BSREAD_DATA_HEADER_COMPRESSION)

        sender.open(no_client_action=no_client_timeout, no_client_timeout=config.MFLOW_NO_CLIENTS_TIMEOUT)
        # TODO: Register proper channels.

        # Indicate that the startup was successful.
        stop_event.clear()

        _logger.debug("Transceiver started. %s", log_tag)
        downsampling_counter = sys.maxsize  # The first is always sent
        last_sent_timestamp = 0
        last_rcvd_timestamp = time.time()

        while not stop_event.is_set():
            try:
                while not parameter_queue.empty():
                    new_parameters = parameter_queue.get()
                    pipeline_config.set_configuration(new_parameters)
                    pipeline_parameters, image_background_array = process_pipeline_parameters()

                data = source.receive()
                set_statistics(statistics, sender, data.statistics.total_bytes_received if data else statistics.total_bytes)
                if data:
                    last_rcvd_timestamp = time.time()
                else:
                    timeout = pipeline_parameters.get("camera_timeout")
                    if timeout:
                        if (timeout > 0) and (time.time() - last_rcvd_timestamp) > timeout:
                            _logger.warning("Camera timeout. %s", log_tag)
                            #Try reconnecting to the camera. If fails raise exception and stops pipeline.
                            connect_to_camera()
                    continue

                # Check downsampling parameter
                downsampling = pipeline_parameters.get("downsampling")
                if downsampling:
                    downsampling_counter += 1
                    if downsampling_counter > downsampling:
                        downsampling_counter = 0
                    else:
                        continue

                #Check maximum frame rate parameter
                max_frame_rate = pipeline_parameters.get("max_frame_rate")
                if max_frame_rate:
                    min_interval = 1.0 / max_frame_rate
                    if (time.time() - last_sent_timestamp) < min_interval:
                        continue

                image = data.data.data["image"].value
                x_axis = data.data.data["x_axis"].value
                y_axis = data.data.data["y_axis"].value

                #Check for rotation parameter
                rotation = pipeline_parameters.get("rotation")
                if rotation:
                    image = rotate(image, rotation["angle"], rotation["order"], rotation["mode"])
                else:
                    # Make a copy if the original image (can be used by multiple pipelines)
                    # image = numpy.array(image)

                    # If image is greater that the huge page size (2MB) then image copy makesCPU consumption increase by orders
                    # of magnitude. Perform a copy in chunks instead, where each chunk is smaller than 2MB
                    image = chunk_copy(image)

                processing_timestamp = data.data.data["timestamp"].value

                function = get_function(pipeline_parameters.get("function"), pipeline_parameters.get("reload"))
                if not function:
                    continue

                processed_data = function(image, processing_timestamp, x_axis, y_axis,
                                               pipeline_parameters, image_background_array)

                processed_data["width"] = processed_data["image"].shape[1]
                processed_data["height"] = processed_data["image"].shape[0]

                # Requesting subset of the data
                include = pipeline_parameters.get("include")
                if include:
                    aux = {}
                    for key in include:
                        aux[key] = processed_data.get(key)
                    processed_data = aux
                exclude = pipeline_parameters.get("exclude")
                if exclude:
                    for field in exclude:
                        processed_data.pop(field, None)

                pulse_id = data.data.pulse_id
                timestamp = (data.data.global_timestamp, data.data.global_timestamp_offset)

                last_sent_timestamp = time.time()
                sender.send(data=processed_data, timestamp=timestamp, pulse_id=pulse_id)

            except:
                _logger.exception("Could not process message. %s",  log_tag)
                stop_event.set()

        _logger.info("Stopping transceiver. %s", log_tag)

    except:
        _logger.exception("Exception while trying to start the receive and process thread. %s", log_tag)
        raise

    finally:
        if source:
            source.disconnect()

        if sender:
            sender.close()


def store_pipeline(stop_event, statistics, parameter_queue,
                   cam_client, pipeline_config, output_stream_port, background_manager):

    def no_client_timeout():
        _logger.warning("No client connected to the pipeline stream for %d seconds. Closing instance. %s",
                        config.MFLOW_NO_CLIENTS_TIMEOUT, log_tag)
        stop_event.set()

    source = None
    sender = None

    try:
        init_statistics(statistics)

        camera_name = pipeline_config.get_camera_name()
        log_tag = " [" + str(camera_name) + " | " + str(pipeline_config.get_name()) + ":" + str(output_stream_port) + "]"
        camera_stream_address = cam_client.get_instance_stream(camera_name)

        _logger.debug("Connecting to camera stream address %s. %s", camera_stream_address, log_tag)

        source_host, source_port = get_host_port_from_stream_address(camera_stream_address)

        source = Source(host=source_host, port=source_port, receive_timeout=config.PIPELINE_RECEIVE_TIMEOUT, mode=SUB)

        source.connect()

        _logger.debug("Opening output stream on port %d. %s", output_stream_port,  log_tag)

        sender = Sender(port=output_stream_port, mode=PUSH,
                        data_header_compression=config.CAMERA_BSREAD_DATA_HEADER_COMPRESSION, block=False)

        sender.open(no_client_action=no_client_timeout, no_client_timeout=config.MFLOW_NO_CLIENTS_TIMEOUT)
        # TODO: Register proper channels.
        # Indicate that the startup was successful.
        stop_event.clear()

        _logger.debug("Transceiver started. %s", log_tag)

        while not stop_event.is_set():
            try:
                data = source.receive()
                set_statistics(statistics, sender, data.statistics.total_bytes_received if data else statistics.total_bytes)

                # In case of receiving error or timeout, the returned data is None.
                if data is None:
                    continue

                forward_data = {camera_name: data.data.data["image"].value}

                pulse_id = data.data.pulse_id
                timestamp = (data.data.global_timestamp, data.data.global_timestamp_offset)

                sender.send(data=forward_data, pulse_id=pulse_id, timestamp=timestamp)

            except:
                _logger.exception("Could not process message. %s", log_tag)
                stop_event.set()

        _logger.info("Stopping transceiver. %s", log_tag)

    except:
        _logger.exception("Exception while trying to start the receive and process thread. %s", log_tag)
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
