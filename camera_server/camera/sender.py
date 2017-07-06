import time
from logging import getLogger

from bsread import bsread

_logger = getLogger(__name__)


class Sender(object):
    def __init__(self, queue_size=10, port=9999, conn_type=bsread.sender.BIND, mode=bsread.sender.PUB, block=True,
                 start_pulse_id=0):
        self.sender = bsread.sender.Sender(queue_size=queue_size, port=port, conn_type=conn_type, mode=mode,
                                           block=block, start_pulse_id=start_pulse_id)

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

    def send(self, data=None, **kwargs):
        if data:
            self.sender.send(pulse_id=0, **data)  # Support for just passing a dict
            pass
        else:
            self.sender.send(pulse_id=0, **kwargs)
            pass

    def close(self):
        self.sender.close()
