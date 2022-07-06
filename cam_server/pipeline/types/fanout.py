from cam_server.pipeline.utils import *
from logging import getLogger
import sys
from collections import OrderedDict
from cam_server.utils import  init_statistics

_logger = getLogger(__name__)


def run(stop_event, statistics, parameter_queue, cam_client, pipeline_config, output_stream_port,
        background_manager, user_scripts_manager=None):

    camera_name = pipeline_config.get_camera_name()
    set_log_tag(" [" + str(camera_name) + " | " + str(pipeline_config.get_name()) + ":" + str(output_stream_port) + "]")
    exit_code = 0


    try:

        init_statistics(statistics)
        init_pipeline_parameters(pipeline_config)
        # Indicate that the startup was successful.
        stop_event.clear()

        connect_to_source(cam_client)
        setup_sender(output_stream_port, stop_event)


        _logger.debug("Transceiver started. %s" % log_tag)

        while not stop_event.is_set():
            try:
                pulse_id, global_timestamp, data = receive_stream()

                if not data:
                    continue

                stream_data = OrderedDict()
                try:
                    for key, value in data.items():
                        stream_data[key] = value.value
                except Exception as e:
                    _logger.error("Error parsing bsread message: " + str(e) + ". %s" % log_tag)
                    continue

                send_data(stream_data, global_timestamp, pulse_id)
            except Exception as e:
                _logger.exception("Could not process message: " + str(e) + ". %s" % log_tag)
                stop_event.set()

    except Exception as e:
        _logger.exception("Exception starting the receive thread: " + str(e) + ". %s" % log_tag)
        exit_code = 1
        raise

    finally:
        _logger.info("Stopping transceiver. %s" % log_tag)
        stop_event.set()
        cleanup()
        _logger.debug("Exiting process. %s" % log_tag)
        sys.exit(exit_code)