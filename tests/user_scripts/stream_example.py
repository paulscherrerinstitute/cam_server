    
from cam_server.pipeline.data_processing import functions


def process(stream_data, pulse_id, timestamp, parameters):
    stream_data["output"] = 10.0
    return stream_data
