import time
import numpy
from collections import OrderedDict

def process(parameters):
    stream_data = OrderedDict()
    stream_data["scalar"] = 10.0
    stream_data["waveform"] = numpy.random.randint(1, 101, 10, "uint16")
    stream_data["image"] = numpy.random.randint(1, 101, 200, "uint16").reshape((10, 20))
    timestamp = time.time()
    pulse_id = None
    data_size = stream_data["image"].size
    return stream_data, timestamp, pulse_id, data_size
