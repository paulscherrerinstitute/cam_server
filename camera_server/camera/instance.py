import multiprocessing
import time
from logging import getLogger

from camera_server.camera.sender import Sender

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
        self.statistics = self.manager.Namespace()
        self.parameter_queue = multiprocessing.Queue()
        self.stream_address = None

    def start(self, parameter, *args):
        if self.process and self.process.is_alive():
            _logger.info("Instance already running")
            return

        if parameter is not None:
            self.parameter_queue.put(parameter)

        self.process = multiprocessing.Process(target=self.process_function,
                                               args=(self.stop_event, self.statistics, self.parameter_queue, *args))
        self.process.start()

    def stop(self):
        if not self.process:
            _logger.info("Instance already stopped")
            return

        self.stop_event.set()

        # Wait maximum of 10 seconds for process to stop
        for i in range(100):
            time.sleep(0.1)
            if not self.process.is_alive():
                break

        # Kill process - no-op in case process already terminated
        self.process.terminate()
        self.process = None

    def wait(self):
        self.process.join()

    def set_parameter(self, parameters):
        self.parameter_queue.put(parameters)

    def is_running(self):
        return self.process and self.process.is_alive()
