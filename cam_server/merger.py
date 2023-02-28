import json
import time
from collections import OrderedDict
from collections import deque
from logging import getLogger
from threading import Thread

from bsread import SUB, DEFAULT_DISPATCHER_URL
from bsread import source as bssource
from bsread.sender import CONNECT

from mflow import Statistics, Message
from bsread.handlers.compact import Message as MessageData


from cam_server import config
from cam_server_client.utils import get_host_port_from_stream_address

_logger = getLogger(__name__)


class StreamSource():
    def __init__(self, address, mode=SUB, queue_size=100, receive_timeout=10):
        self.channel_list=None
        self.mode=mode
        self.address = address
        self.source = None
        self.dispatcher_url = DEFAULT_DISPATCHER_URL
        self.dispatcher_verify_request=True
        self.dispatcher_disable_compression = False
        self.queue_size = queue_size
        self.receive_timeout = receive_timeout

    def connect(self):
        if self.address:
            self.host, self.port = get_host_port_from_stream_address(self.address)
        else:
            self.host, self.port = None, 9999

        if self.channel_list is not None:
            if type(self.channel_list) != list:
                self.channel_list = json.loads(self.channel_list)
            if len(self.channel_list) == 0:
                self.channel_list = None

        ret = bssource(host=self.host,
                       conn_type=CONNECT,
                       port=self.port,
                       mode=self.mode,
                       channels=self.channel_list,
                       queue_size=self.queue_size,
                       receive_timeout=self.receive_timeout,
                       dispatcher_url=self.dispatcher_url,
                       dispatcher_verify_request=self.dispatcher_verify_request,
                       dispatcher_disable_compression=self.dispatcher_disable_compression)
        self.source = ret.source
        self.source.connect()
        return self.source

    def disconnect(self):
        if self.source:
            try:
                self.source.disconnect()
            except:
                pass
            finally:
                self.source = None
    def receive(self):
        if self.source:
            try:
                rx = self.source.receive()
                if rx:
                    msg=rx.data
                    stats=rx.statistics
                    stats.total_bytes_received
                    pulse_id = msg.pulse_id
                    global_timestamp = (msg.global_timestamp, msg.global_timestamp_offset)
                    data = msg.data
                    return pulse_id, global_timestamp, data, stats
            except:
                pass
        return None, None, None, None


class DispatcherSource(StreamSource):
    # dispatcher_url, dispatcher_verify_request, dispatcher_disable_compression = get_dispatcher_parameters(self.pars)
    def __init__(self, channel_list, dispatcher_url=DEFAULT_DISPATCHER_URL, dispatcher_verify_request=True, dispatcher_disable_compression=False, queue_size=100, receive_timeout=10):
        self.channel_list=channel_list
        self.mode=SUB
        self.address = None
        self.source = None
        self.dispatcher_url = dispatcher_url
        self.dispatcher_verify_request=dispatcher_verify_request
        self.dispatcher_disable_compression = dispatcher_disable_compression
        self.queue_size = queue_size
        self.receive_timeout = receive_timeout

class Merger():
    def __init__(self, stream1, stream2, receive_timeout=config.PIPELINE_RECEIVE_TIMEOUT, message_buffer_size=100):
        self.stream1, self.stream2 = stream1, stream2
        self.source1, self.source2 = None, None
        self.connected = False
        self.processing_thread = None
        self.receive_timeout = receive_timeout
        self.message_buffer = deque(maxlen=message_buffer_size)
        self.statistics = Statistics()

    def run(self):
        pulse_id1, pulse_id2= None, None
        try:
            _logger.info("Entering merger thread")
            self.connect_sources()
            #start = time.time()
            #self.lastpid=0
            while self.is_connected():
                if (pulse_id1 is None) or ((pulse_id1 is not None) and (pulse_id2 is not None) and (pulse_id1<pulse_id2)):
                    pulse_id1, global_timestamp1, data1, stats1 = self.stream1.receive()
                    #if pulse_id1: print ("Rec 1 ", pulse_id1, time.time()-start)
                if (pulse_id2 is None) or ((pulse_id1 is not None) and (pulse_id2 is not None) and (pulse_id2<pulse_id1)):
                    pulse_id2, global_timestamp2, data2, stats2 = self.stream2.receive()
                    #if pulse_id2: print ("Rec 2 ", pulse_id2, time.time()-start)
                if (pulse_id1 is not None) and (pulse_id2 is not None) and (pulse_id1 == pulse_id2):
                    pulse_id, global_timestamp = pulse_id1, global_timestamp1
                    pulse_id1, pulse_id2 = None, None
                    try:
                        data = OrderedDict()
                        if data1:
                            data.update(data1)
                        if data2:
                            data.update(data2)
                    except Exception as e:
                        _logger.warning("Cannot merge pulse id : " + pulse_id1 + " - " + str(e))
                        continue
                    #print("Merge ", pulse_id, time.time()-start)
                    #if self.lastpid:
                    #    if(self.lastpid+1) != pulse_id:
                    #            print ("Expecting  ", (self.lastpid+1), " sent ", pulse_id)
                    #self.lastpid = pulse_id
                    self.on_receive_data(pulse_id, global_timestamp, data, stats1, stats2)
        except Exception as e:
            _logger.exception("Error in merger: " + str(e))
            self.connected = False
        finally:
            self.disconnect_sources()
            _logger.info("Exiting merger thread")

    def connect_sources(self):
        try:
            self.source1 = self.stream1.connect()
            self.source2 = self.stream2.connect()
        except:
            self.disconnect_sources()
            raise


    def disconnect_sources(self):
        if self.source1:
            self.source1.disconnect()
            self.source1 = None
        if self.source2:
            self.source2.disconnect()
            self.source2 = None

    def on_receive_data(self, pulse_id, global_timestamp, data, stats1, stats2):
        self.message_buffer.append((pulse_id, global_timestamp, data, stats1, stats2))

    #Source interface
    def receive(self):
      try:
        timeout = time.time() + (float(self.receive_timeout)/1000)
        while True:
            if len(self.message_buffer) > 0:
                ret = (pulse_id, global_timestamp, data, stats1, stats2) = self.message_buffer.popleft()
                data = MessageData(pulse_id, global_timestamp[0], global_timestamp[1],  data=data)
                self.statistics.bytes_received = stats1.bytes_received + stats2.bytes_received
                self.statistics.total_bytes_received = self.statistics.total_bytes_received + self.statistics.bytes_received
                self.statistics.messages_received = self.statistics.messages_received + 1
                return Message(self.statistics, data)
            if time.time()>timeout:
                return None
            time.sleep(0.001)
      except Exception as e:
          _logger.exception("Error in receive: " + str(e))

    def connect(self):
        self.disconnect()
        self.connected = True
        self.processing_thread = Thread(target=self.run, args=())
        self.processing_thread.start()

    def disconnect(self):
        self.connected = False
        if self.processing_thread:
            try:
                self.processing_thread.join(1.0)
            except:
                pass
            self.processing_thread = None

    def is_connected(self):
        return self.connected

    #Context manager interface
    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.disconnect()

if __name__ == '__main__':
    st1 = StreamSource("tcp://localhost:5554")
    st2 = StreamSource("tcp://localhost:5552")
    with Merger(st1, st2, 10) as m:
        m.connect()
        time.sleep(5.0)
        while True:
            r=m.receive()
            if not r:
                break
            print(r.data.pulse_id)
        m.disconnect()