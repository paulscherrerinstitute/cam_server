from logging import getLogger
from bsread.sender import Sender
from zmq import Again

from cam_server import config

_logger = getLogger(__name__)


def process_camera_stream(stop_event, statistics, parameter_queue,
                          camera, port):
    """
    Start the camera stream and listen for image monitors. This function blocks until stop_event is set.
    :param stop_event: Event when to stop the process.
    :param statistics: Statistics namespace.
    :param camera: Camera instance to get the images from.
    :param port: Port to use to bind the output stream.
    """

    # If there is no client for some time, disconnect.
    def no_client_timeout():
        _logger.info("No client connected to the '%s' stream for %d seconds. Closing instance." %
                     (camera.get_name(), config.MFLOW_NO_CLIENTS_TIMEOUT))
        stop_event.set()

    sender = Sender(port=port, send_timeout=config.CAMERA_SEND_TIMEOUT)

    # Register the bsread channels - compress only the image.
    sender.add_channel("image", metadata={"compression": config.CAMERA_BSREAD_IMAGE_COMPRESSION})
    sender.add_channel("timestamp", metadata={"compression": None})

    sender.open(no_client_action=no_client_timeout, no_client_timeout=config.MFLOW_NO_CLIENTS_TIMEOUT)

    camera.connect()

    statistics.counter = 0

    def collect_and_send(image, timestamp):
        # Data to be sent over the stream.
        data = {"image": image,
                "timestamp": timestamp}

        try:
            sender.send(data=data, check_data=False)
        except Again:
            _logger.warning("Send timeout. Lost image with timestamp '%s'." % timestamp)

    camera.add_callback(collect_and_send)

    # This signals that the camera has successfully started.
    stop_event.clear()

    # Wait for termination / update configuration / etc.
    stop_event.wait()

    camera.disconnect()
    sender.close()
