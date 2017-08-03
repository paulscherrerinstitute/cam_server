from logging import getLogger

import time
from bsread import Source, PUB
from bsread.sender import Sender

from cam_server import config, CamClient
from cam_server.pipeline.data_processing.processor import process_image
from cam_server.utils import get_host_port_from_stream_address
from zmq import Again

_logger = getLogger(__name__)


def receive_process_send(stop_event, statistics, parameter_queue,
                         cam_api_endpoint, pipeline_config, output_stream_port, background_manager):
    def no_client_timeout():
        _logger.warning("No client connected to the pipeline stream for %d seconds. Closing instance." %
                        config.MFLOW_NO_CLIENTS_TIMEOUT)
        stop_event.set()
    try:

        client = CamClient(cam_api_endpoint)
        camera_stream_address = client.get_camera_stream(pipeline_config.get_camera_name())
        source_host, source_port = get_host_port_from_stream_address(camera_stream_address)

        source = Source(host=source_host, port=source_port, receive_timeout=config.PIPELINE_RECEIVE_TIMEOUT)
        source.connect()

        sender = Sender(port=output_stream_port, send_timeout=config.CAMERA_SEND_TIMEOUT, mode=PUB)
        sender.open(no_client_action=no_client_timeout, no_client_timeout=config.MFLOW_NO_CLIENTS_TIMEOUT)

        pipeline_parameters = pipeline_config.get_parameters()
        image_background_array = background_manager.get_background(pipeline_config.get_background_id())

        # TODO: Register proper channels.

        while not stop_event.is_set():
            try:
                while not parameter_queue.empty():
                    pipeline_config.parameters = parameter_queue.get()
                try:
                    data = source.receive()
                except Again:
                    _logger.error("Source '%s:%d' did not provide message in time. Aborting.", source_host, source_port)
                    stop_event.set()

                image = data.data.data["image"].value
                timestamp = data.data.data["timestamp"].value
                x_axis = data.data.data["x_axis"].value
                y_axis = data.data.data["y_axis"].value

                processed_data = process_image(image, timestamp, x_axis, y_axis,
                                               pipeline_parameters, image_background_array)

                sender.send(data=processed_data)

            except:
                _logger.exception("Could not process message.")
                stop_event.set()

        source.disconnect()
        sender.close()

    except:
        _logger.exception("Exception while trying to start the receive and process thread.")
