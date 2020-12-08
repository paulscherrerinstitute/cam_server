from bsread.bsread import *
from bsread.sender import *


class IpcSource(Source):
    def __init__(self, address, config_port=None, conn_type=CONNECT, mode=None, queue_size=100,
                 copy=True, receive_timeout=None):
        self.use_dispatching_layer = False
        self.address = address
        self.host = None
        self.port = -1
        self.config_port = config_port
        self.conn_type = conn_type
        self.queue_size = queue_size
        self.copy = copy
        self.receive_timeout = receive_timeout
        self.mode = mode if mode else PULL
        self.stream = None
        self.handler = Handler()


class IpcSender(Sender):
    def open(self, no_client_action=None, no_client_timeout=None):
        self.stream = mflow.connect(self.address, queue_size=self.queue_size,
                                    conn_type=self.conn_type, mode=self.mode, no_client_action=no_client_action,
                                    no_client_timeout=no_client_timeout, copy=self.copy, send_timeout=self.send_timeout)

        # Main header
        self.main_header = dict()
        self.main_header['htype'] = "bsr_m-1.1"
        if self.data_header_compression:
            self.main_header['dh_compression'] = self.data_header_compression

        # Data header
        with self.channels_lock:
            self._create_data_header()

        # Set initial pulse_id
        self.pulse_id = self.start_pulse_id

        # Update internal status
        self.status_stream_open = True


# Support of "with" statement
class ipc_source:

    def __init__(self, address, config_port=None, conn_type=CONNECT, mode=None, queue_size=100,
                 copy=True, receive_timeout=None):
        self.source = IpcSource(address, config_port, conn_type, mode,  queue_size, copy, receive_timeout)

    def __enter__(self):
        self.source.connect()
        return self.source

    def __exit__(self, type, value, traceback):
        self.source.disconnect()
