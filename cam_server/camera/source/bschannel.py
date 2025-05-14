from bsread import PULL, SUB, Source

from cam_server.camera.source.camera import *
from cam_server.utils import get_host_port_from_stream_address

_logger = getLogger(__name__)


class BsreadChannel(Camera):
    def __init__(self, camera_config):
        super(BsreadChannel, self).__init__(camera_config)
        self.camera_config = camera_config
        self.input_stream_address = None
        self.source = None
        self.channel = None
        self.mode = None
        self.receive_timeout = config.ZMQ_RECEIVE_TIMEOUT
        self.timestamp = None
        self.pulse_id = None
        self.height_raw, self.width_raw = 0,0
        self.input_stream_address = None

    def connect(self):
        self.timestamp = None
        self.pulse_id = None
        if not self.input_stream_address:
            self.input_stream_address = self.camera_config.get_source()
        source_host, source_port = get_host_port_from_stream_address(self.input_stream_address)
        self.mode = self.camera_config.get_configuration().get("mode", SUB)
        self.channel = self.camera_config.get_configuration().get("channel", None)
        self.source = Source(host=source_host, port=source_port, mode=self.mode, receive_timeout=self.receive_timeout)
        self.source.connect()
        #Try not read size not to consume first message

    def disconnect(self):
        try:
            if self.source is not None:
                self.source.disconnect()
        except:
            pass
        finally:
            self.source = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

    def read(self):
        try:
            msg = self.source.receive()
            if not msg:
                raise Exception("Received None message.")

            if self.channel is None:
                if len(msg.data.data.keys()) == 1:
                    self.channel = msg.data.data.keys()[0]
                else:
                    raise Exception("Undefined channel")
            data = msg.data.data.get(self.channel, None)
            if not data:
                raise Exception("Channel not present in messsage")

            image = data.value
            self.height_raw, self.width_raw = image.shape
            self.dtype = image.dtype
            self.timestamp = (msg.data.global_timestamp, msg.data.global_timestamp_offset)
            self.pulse_id = msg.data.pulse_id
            format_changed = msg.data.format_changed
            return image

        except Exception as e:
            _logger.warning("Error processing bsread stream: %s" % (str(e),))
            raise

    def get_pulse_id(self):
        self.pulse_id

    def get_timestamp(self):
        self.timestamp

