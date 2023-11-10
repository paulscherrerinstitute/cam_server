import sys
from collections import OrderedDict
from logging import getLogger

from cam_server.pipeline.utils import *
from cam_server.utils import init_statistics

_logger = getLogger(__name__)


def run(stop_event, statistics, parameter_queue, logs_queue, cam_client, pipeline_config, output_stream_port,
        background_manager, user_scripts_manager=None):

    exit_code = 0

    init_pipeline_parameters(pipeline_config, parameter_queue, logs_queue, user_scripts_manager, port=output_stream_port)


    def process_stream(pulse_id, global_tamestamp, function,input_data):
        try:
            return function(input_data, pulse_id, global_tamestamp, get_parameters())
        except Exception as e:
            #import traceback
            #traceback.print_exc()
            _logger.warning("Error processing PID %d at proc %d thread %d: %s" % (pulse_id, os.getpid(), threading.get_ident(), str(e)))
            if abort_on_error():
                raise

    try:

        init_statistics(statistics)

        # Indicate that the startup was successful.
        stop_event.clear()

        setup_sender(output_stream_port, stop_event, process_stream, user_scripts_manager)


        _logger.debug("Transceiver started. %s" % log_tag)

        with connect_to_stream():
            while not stop_event.is_set():
                try:
                    check_parameters_changes()
                    assert_function_defined()

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

                    process_data(process_stream, pulse_id, global_timestamp, stream_data)

                except Exception as e:
                    _logger.error("Error receiving bsread message: " + str(e) + ". %s" % log_tag)
                    _logger.exception("Could not process message: " + str(e) + ". %s" % log_tag)
                    stop_event.set()

    except Exception as e:
        _logger.exception("Exception starting the receive thread: " + str(e) + ". %s" % log_tag)
        exit_code = 1
        raise

    finally:
        stop_event.set()
        cleanup(exit_code)
