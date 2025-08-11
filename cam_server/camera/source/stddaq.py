from cam_server.camera.source.bschannel import BsreadChannel
from cam_server.utils import get_host_port_from_stream_address
from logging import getLogger
try:
    import redis
except:
    redis = None

_logger = getLogger(__name__)


class StdDaq(BsreadChannel):
    def __init__(self, camera_config):
        super(StdDaq, self).__init__(camera_config)
        if redis is None:
            raise Exception("Redis is not installed")
        self.url = camera_config.get_configuration().get("url", "sf-daq-6.psi.ch:6379")
        self.host, self.port = get_host_port_from_stream_address(self.url)
        self.db = camera_config.get_configuration().get("db", '0')
        self.device = camera_config.get_source()

    def connect(self):
        self.input_stream_address = self.get_instance_stream(self.device)
        self.channel = self.camera_config.get_configuration().get("channel", self.device + ":FPICTURE")
        super(StdDaq, self).connect()

    def get_instance_stream(self, name):
        with redis.Redis(host=self.host, port=self.port, db=self.db) as r:
            ret = r.get(name)
            return ret.decode('utf-8').strip() if ret else ret

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()


