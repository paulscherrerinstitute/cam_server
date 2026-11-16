import sys
from collections import OrderedDict
from logging import getLogger

from cam_server.pipeline.utils import *
from cam_server.utils import init_statistics

_logger = getLogger(__name__)


def run(stop_event, statistics, parameter_queue, logs_queue,cam_client, pipeline_config, output_stream_port,
        background_manager, user_scripts_manager=None):

    exit_code = 0

    try:
        init_statistics(statistics)
        init_pipeline_parameters(pipeline_config, port=output_stream_port, logs_queue=logs_queue)
        # Indicate that the startup was successful.
        stop_event.clear()

        connect_to_source(cam_client)
        setup_sender(output_stream_port, stop_event)

        #Reorder parameters
        pars = get_parameters()
        reorder = pars.get("reorder", False)
        MAX_BUFFER = pars.get("reorder_buffer", 100)
        GAP_TIMEOUT = pars.get("reorder_timeout", 1.0)
        buffer = {}
        next_expected_id = None
        gap_since = None

        _logger.info("Transceiver started.%s" % get_log_tag())

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
                    _logger.error("Error parsing bsread message: %s.%s" % (str(e), get_log_tag()))
                    continue

                if not reorder:
                    send_data(stream_data, global_timestamp, pulse_id)
                else:
                    if next_expected_id is None:
                        next_expected_id = pulse_id
                    if pulse_id >= next_expected_id:
                        buffer.setdefault(pulse_id, (global_timestamp, stream_data))

                    # Send all currently available consecutive messages.
                    while next_expected_id in buffer:
                        timestamp, stream_data = buffer.pop(next_expected_id)
                        send_data(stream_data, timestamp, next_expected_id)
                        next_expected_id += 1

                    # Detect a gap in the sequence.
                    if buffer and next_expected_id not in buffer:
                        if gap_since is None:
                            gap_since = time.monotonic()

                        gap_expired = time.monotonic() - gap_since >= GAP_TIMEOUT
                        buffer_full = len(buffer) >= MAX_BUFFER

                        if gap_expired or buffer_full:
                            smallest_id = min(buffer)
                            if debug:
                                _logger.info("Fan-in pulse gap: expected %s, next available %s buffer size %s. Skipping missing IDs.", next_expected_id, smallest_id, len(buffer))
                            next_expected_id = smallest_id
                            gap_since = None
                    else:
                        gap_since = None

            except Exception as e:
                _logger.exception("Exception trying to start the receive thread: %s.%s" % (str(e), get_log_tag()))
                stop_event.set()

    except Exception as e:
        _logger.exception("Exception trying to start the receive thread: %s.%s" % (str(e), get_log_tag()))
        exit_code = 1
        raise

    finally:
        stop_event.set()
        cleanup(exit_code)
