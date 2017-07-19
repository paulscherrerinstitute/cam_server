import multiprocessing
import time
from logging import getLogger

from cam_server import config

_logger = getLogger(__name__)


class CameraInstance:
    def __init__(self, process_function, camera, stream_port):

        self.process_function = process_function
        self.camera = camera
        self.stream_port = stream_port

        self.process = None
        self.manager = multiprocessing.Manager()

        self.stop_event = multiprocessing.Event()
        # The initial value of the stop event is set -> when the process starts, it un-sets it to signal the start.
        self.stop_event.set()

        self.statistics = self.manager.Namespace()

        # TODO: Retrieve real address.
        self.stream_address = "tcp://%s:%d" % ("127.0.0.1", self.stream_port)

    def start(self):
        if self.process and self.process.is_alive():
            _logger.info("Camera instance '%s' already running.", self.camera_name)
            return

        self.process = multiprocessing.Process(target=self.process_function,
                                               args=(self.stop_event, self.statistics,
                                                     self.camera, self.stream_port))
        self.process.start()

        # Wait for the processor to clear the flag - indication that the process is ready.
        start_timestamp = time.time()
        while self.stop_event.is_set():
            time.sleep(config.PROCESS_POLL_INTERVAL)
            # Check if the timeout has already elapsed.
            if time.time() - start_timestamp > config.PROCESS_COMMUNICATION_TIMEOUT:
                self.process.terminate()
                error_message = "Could not start the '%s' camera in time. Terminated. See cam_server logs." % \
                                self.camera.prefix
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

    def get_info(self):
        return {"stream_address": self.stream_address,
                "is_stream_active": self.is_running(),
                "camera_geometry": self.camera.get_geometry(),
                "camera_prefix": self.camera.prefix}

