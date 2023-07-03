import logging
from cam_server.pipeline.data_processing import functions

_logger = logging.getLogger(__name__)

def process(stream_data, pulse_id, timestamp, parameters):
    stream_data["output"] = 10.0    
    _logger.info("Success processing stream_example")
    return stream_data
