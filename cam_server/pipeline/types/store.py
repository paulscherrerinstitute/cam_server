import sys
from logging import getLogger

from bsread import PUSH
from bsread.sender import Sender

from cam_server import config
from cam_server.pipeline.utils import *
from cam_server.utils import update_statistics, init_statistics

_logger = getLogger(__name__)

def run(stop_event, statistics, parameter_queue, logs_queue, cam_client, pipeline_config, output_stream_port,
        background_manager, user_scripts_manager=None):

    def no_client_action():
        nonlocal  parameters
        if parameters["no_client_timeout"] > 0:
            _logger.warning("No client connected to the pipeline stream for %d seconds. Closing instance. %s" %
                        (config.MFLOW_NO_CLIENTS_TIMEOUT, get_log_tag()))
            stop_event.set()

    source = None
    sender = None
    exit_code = 0

    parameters = init_pipeline_parameters(pipeline_config, parameter_queue, logs_queue, user_scripts_manager, port=output_stream_port)
    if parameters.get("no_client_timeout") is None:
        parameters["no_client_timeout"] = config.MFLOW_NO_CLIENTS_TIMEOUT
    modulo = parameters.get("modulo", None)

    try:
        init_statistics(statistics)

        camera_name = pipeline_config.get_camera_name()
        stream_image_name = camera_name + config.EPICS_PV_SUFFIX_IMAGE

        source = connect_to_camera(cam_client)

        _logger.debug("Opening output stream on port %d. %s", output_stream_port,  get_log_tag())

        sender = Sender(port=output_stream_port, mode=PUSH,
                        data_header_compression=config.CAMERA_BSREAD_DATA_HEADER_COMPRESSION, block=False)
        sender.open(no_client_action=no_client_action, no_client_timeout=parameters["no_client_timeout"]
                    if parameters["no_client_timeout"] > 0 else sys.maxsize)
        init_sender(sender, parameters)

        # TODO: Register proper channels.
        # Indicate that the startup was successful.
        stop_event.clear()

        _logger.info("Transceiver started. %s" % get_log_tag())
        counter = 1

        while not stop_event.is_set():
            try:
                data = source.receive()
                update_statistics(sender, data.statistics.total_bytes_received if data else 0, 1 if data else 0)

                if modulo:
                    if counter < modulo:
                        counter = counter + 1
                        continue
                    else:
                        counter = 1

                # In case of receiving error or timeout, the returned data is None.
                if data is None:
                    continue

                forward_data = {stream_image_name: data.data.data["image"].value}

                pulse_id = data.data.pulse_id
                timestamp = (data.data.global_timestamp, data.data.global_timestamp_offset)

                send(sender, forward_data, timestamp, pulse_id)

            except:
                _logger.exception("Error in pipeline processing: %s. %s" % (str(e), get_log_tag()))
                stop_event.set()

        _logger.info("Stopping transceiver. %s" % get_log_tag())

    except Exception as e:
        _logger.exception("Exception trying to start the receive thread: %s. %s" % (str(e), get_log_tag()))
        exit_code = 1
        raise

    finally:
        _logger.info("Stopping transceiver. %s" % get_log_tag())
        if source:
            source.disconnect()

        if sender:
            try:
                sender.close()
            except:
                pass
        sys.exit(exit_code)
