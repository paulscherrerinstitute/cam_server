import multiprocessing
import time
from logging import getLogger

_logger = getLogger(__name__)

# Time to wait for the process to execute the requested action.
PROCESS_COMMUNICATION_TIMEOUT = 3
# Interval used when polling the state from the process.
PROCESS_POLL_INTERVAL = 0.1


class InstanceWrapper:
    def __init__(self, instance_name, process_function, *args):

        self.instance_name = instance_name
        self.process_function = process_function
        # Arguments to pass to the process function.
        self.args = args

        self.process = None
        self.manager = multiprocessing.Manager()

        self.stop_event = multiprocessing.Event()
        # The initial value of the stop event is set -> when the process starts, it un-sets it to signal the start.
        self.stop_event.set()

        self.statistics = self.manager.Namespace()
        self.parameter_queue = multiprocessing.Queue()

    def start(self):
        if self.process and self.process.is_alive():
            _logger.info("Instance '%s' already running.", self.instance_name)
            return

        self.process = multiprocessing.Process(target=self.process_function,
                                               args=(self.stop_event, self.statistics, self.parameter_queue,
                                                     *self.args))
        self.process.start()

        # Wait for the processor to clear the flag - indication that the process is ready.
        start_timestamp = time.time()
        while self.stop_event.is_set():
            time.sleep(PROCESS_POLL_INTERVAL)
            # Check if the timeout has already elapsed.
            if time.time() - start_timestamp > PROCESS_COMMUNICATION_TIMEOUT:
                self.process.terminate()
                error_message = "Could not start the '%s' camera in time. Terminated. See cam_server logs." % \
                                self.instance_name
                _logger.error(error_message)
                raise Exception(error_message)

    def stop(self):
        if not self.process:
            _logger.info("Instance '%s' already stopped.", self.instance_name)
            return

        self.stop_event.set()

        # Wait for the processor to clear the flag - indication that the process is ready.
        start_timestamp = time.time()

        while self.process.is_alive():
            time.sleep(PROCESS_POLL_INTERVAL)
            # Check if the timeout has already elapsed.
            if time.time() - start_timestamp > PROCESS_COMMUNICATION_TIMEOUT:
                _logger.warning("Could not stop the '%s' camera in time. Terminated.", self.instance_name)

        # Kill process - no-op in case process already terminated
        self.process.terminate()
        self.process = None

    def wait(self):
        self.process.join()

    def set_parameter(self, parameters):
        self.parameter_queue.put(parameters)

    def is_running(self):
        return self.process and self.process.is_alive()
