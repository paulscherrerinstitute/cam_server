from logging import getLogger

from cam_server.pipeline.utils import *
from cam_server.utils import update_statistics, init_statistics

_logger = getLogger(__name__)


def run(stop_event, statistics, parameter_queue, cam_client, pipeline_config, output_stream_port,
        background_manager, user_scripts_manager=None):

    exit_code = 0

    init_pipeline_parameters(pipeline_config, parameter_queue, user_scripts_manager, port=output_stream_port)
    try:
        init_statistics(statistics)
        create_sender(output_stream_port, stop_event)

        function = get_function(get_parameters(), user_scripts_manager)
        if function is None:
            raise Exception ("Invalid function")
        max_frame_rate = get_parameters().get("max_frame_rate")
        sample_interval = (1.0 / max_frame_rate) if max_frame_rate else None

        _logger.debug("Transceiver started. %s" % log_tag)
        # Indicate that the startup was successful.
        stop_event.clear()
        init=True

        while not stop_event.is_set():
            try:
                if sample_interval:
                    start = time.time()
                check_parameters_changes()

                stream_data, timestamp, pulse_id, data_size = function(get_parameters(), init)
                init = False
                update_statistics(sender,-data_size, 1 if stream_data else 0)

                if not stream_data or stop_event.is_set():
                    continue

                send(sender, stream_data, timestamp, pulse_id)
                if sample_interval:
                    sleep = sample_interval - (time.time()-start)
                    if (sleep>0):
                        time.sleep(sleep)

            except Exception as e:
                _logger.exception("Could not process message: " + str(e) + ". %s" % log_tag)
                if abort_on_error():
                    stop_event.set()

        _logger.info("Stopping transceiver. %s" % log_tag)

    except:
        _logger.exception("Exception while trying to start the receive and process thread. %s" % log_tag)
        exit_code = 1
        raise

    finally:
        _logger.info("Stopping transceiver. %s" % log_tag)
        cleanup()
        sys.exit(exit_code)
