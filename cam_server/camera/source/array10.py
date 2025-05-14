import zmq
from cam_server.camera.source.camera import *
import json

_logger = getLogger(__name__)


class Array10(Camera):
    def __init__(self, camera_config):
        super(Array10, self).__init__(camera_config)
        self.camera_config = camera_config
        self.input_stream_address = None
        self.mode = None
        self.ctx = None
        self.receiver = None
        self.pid = 0

    def connect(self):
        self.input_stream_address = self.camera_config.get_source()
        self.mode = PULL if self.camera_config.get_configuration().get("mode", "SUB") == "PULL" else SUB
        self.ctx = zmq.Context()
        self.receiver = self.ctx.socket(self.mode)
        self.receiver.connect(self.input_stream_address)
        if self.mode == zmq.SUB:
            self.receiver.subscribe("")
        self.receive()
        self.message_count = 0

    def disconnect(self):
        try:
            self.receiver.close()
        except:
            pass
        try:
            self.ctx.term()
        except:
            pass
        finally:
            self.ctx = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

    def receive(self):
        try:
            header = self.receiver.recv()
            self.header = json.loads(''.join(chr(i) for i in header))
            self.height_raw, self.width_raw = self.header.get("shape")
            self.dtype = self.header.get("type", "int8")
            data = self.receiver.recv()
            if data is not None:
                array = numpy.frombuffer(data, dtype=self.dtype)
                self.pid = self.pid + 1
                return array
        except Exception as e:
            _logger.warning("Error processing Array10: %s" % (str(e),))
            raise

    def get_pulse_id(self):
        return self.pid

    def read(self):
        return self.receive()
