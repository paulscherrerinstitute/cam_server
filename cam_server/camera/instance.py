import multiprocessing
import time
from logging import getLogger

from cam_server import config
from cam_server.camera.sender import Sender

_logger = getLogger(__name__)


def process_camera_stream(stop_event, statistics, parameter_queue, camera, port):

    sender = Sender(port=port)
    sender.open()

    camera.connect()
    x_axis, y_axis = camera.get_x_y_axis()

    statistics.counter = 0

    collector = pipeline_server.Collector(parameter_queue, x_axis, y_axis, consumer=sender.send,
                                          number_of_images=number_of_images, stop_event=stop_event)

    camera.add_callback(collector.collect)

    # This signals that the camera has successfully started.
    stop_event.clear()

    # Wait for termination / update configuration / etc.
    stop_event.wait()
    camera.clear_callbacks()

    camera.disconnect()
    sender.close()


class CameraInstance:
    def __init__(self, process_function, camera_name=None):

        self.process_function = process_function
        self.camera_name = camera_name

        self.process = None
        self.manager = multiprocessing.Manager()

        self.stop_event = multiprocessing.Event()
        # The initial value of the stop event is set -> when the process starts, it un-sets it to signal the start.
        self.stop_event.set()

        self.statistics = self.manager.Namespace()
        self.parameter_queue = multiprocessing.Queue()
        self.stream_address = None

    def start(self, parameter=None, *args):
        if self.process and self.process.is_alive():
            _logger.info("Camera instance '%s' already running.", self.camera_name)
            return

        if parameter is not None:
            self.parameter_queue.put(parameter)

        self.process = multiprocessing.Process(target=self.process_function,
                                               args=(self.stop_event, self.statistics, self.parameter_queue, *args))
        self.process.start()

        # Wait for the processor to clear the flag - indication that the process is ready.
        start_timestamp = time.time()
        while self.stop_event.is_set():
            time.sleep(config.PROCESS_POLL_INTERVAL)
            # Check if the timeout has already elapsed.
            if time.time() - start_timestamp > config.PROCESS_COMMUNICATION_TIMEOUT:
                self.process.terminate()
                error_message = "Could not start the '%s' camera in time. Terminated. See cam_server logs." % \
                                self.camera_name
                _logger.error(error_message)
                raise Exception(error_message)

    def stop(self):
        if not self.process:
            _logger.info("Camera instance '%s' already stopped.", self.camera_name)
            return

        self.stop_event.set()

        # Wait for the processor to clear the flag - indication that the process is ready.
        start_timestamp = time.time()

        while self.process.is_alive():
            time.sleep(config.PROCESS_POLL_INTERVAL)
            # Check if the timeout has already elapsed.
            if time.time() - start_timestamp > config.PROCESS_COMMUNICATION_TIMEOUT:
                _logger.warning("Could not stop the '%s' camera in time. Terminated.", self.camera_name)

        # Kill process - no-op in case process already terminated
        self.process.terminate()
        self.process = None

    def wait(self):
        self.process.join()

    def set_parameter(self, parameters):
        self.parameter_queue.put(parameters)

    def is_running(self):
        return self.process and self.process.is_alive()
