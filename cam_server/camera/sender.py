import time
from logging import getLogger

from bsread import BIND, PUSH, sender

from cam_server import config

_logger = getLogger(__name__)


class Sender(object):
    """
    Helper object to simplify the interaction with bsread.
    """
    def __init__(self, queue_size=10, port=9999, conn_type=BIND, mode=PUSH, block=True,
                 start_pulse_id=0):

        self.sender = sender.Sender(queue_size=queue_size, port=port, conn_type=conn_type, mode=mode,
                                    block=block, start_pulse_id=start_pulse_id,
                                    data_header_compression=config.BSREAD_DATA_HEADER_COMPRESSION)

        # Register the bsread channels - compress only the image.
        self.sender.add_channel("image", metadata={"compression": config.BSREAD_IMAGE_COMPRESSION})
        self.sender.add_channel("timestamp", metadata={"compression": None})

    def open(self):
        exception = None
        # Sometimes on Linux binding to a port fails although the port was probed to be free before.
        # Eventually this has to do with the os not releasing the port (port was bind to for the free probe) in time.
        for unused in range(10):
            try:
                self.sender.open()
                break
            except Exception as e:
                _logger.info("Unable to bind to port %d" % self.sender.port)
                exception = e
                time.sleep(1)

        if exception is not None:
            raise exception

    def send(self, data):
        # Speed up - do not need to check data, since we set the channels correctly.
        self.sender.send(data=data, check_data=False)

    def close(self):
        self.sender.close()


def process_camera_stream(stop_event, statistics, camera, port):
    """
    Start the camera stream and listen for image monitors. This function blocks until stop_event is set.
    :param stop_event: Event when to stop the process.
    :param statistics: Statistics namespace.
    :param camera: Camera instance to get the images from.
    :param port: Port to use to bind the output stream.
    """
    stream = Sender(port=port)
    stream.open()

    camera.connect()

    statistics.counter = 0

    def collect_and_send(image, timestamp):
        # Data to be sent over the stream.
        data = {"image": image,
                "timestamp": timestamp}

        stream.send(data)

    camera.add_callback(collect_and_send)

    # This signals that the camera has successfully started.
    stop_event.clear()

    # Wait for termination / update configuration / etc.
    stop_event.wait()

    camera.clear_callbacks()
    camera.disconnect()

    stream.close()
