from logging import getLogger

from bsread import Source
from bsread.sender import Sender

from cam_server import config

_logger = getLogger(__name__)


def receive_process_send(stop_event, statistics, parameter_queue,
                         source_host, source_port, pipeline, output_stream_port):

    def no_client_timeout():
        _logger.info("No client connected to the pipeline stream for %d seconds. Closing instance." %
                     config.MFLOW_NO_CLIENTS_TIMEOUT)
        stop_event.set()

    source = Source(host=source_host, port=source_port, receive_timeout=config.PIPELINE_RECEIVE_TIMEOUT)
    source.connect()

    sender = Sender(port=output_stream_port)
    sender.open(no_client_action=no_client_timeout, no_client_timeout=config.MFLOW_NO_CLIENTS_TIMEOUT)
    # TODO: Register proper channels.
    # TODO: Add configuration hash.

    while not stop_event.is_set():
        while not parameter_queue.empty():
            pipeline.parameters = parameter_queue.get()

        data = source.receiver()

        if data is not None:
            processed_data = pipeline(data)
            sender.send(data=processed_data)

    source.disconnect()
    sender.close()
