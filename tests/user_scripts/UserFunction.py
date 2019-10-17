import json
from logging import getLogger
import numpy
from cam_server.pipeline.data_processing import functions, processor
_logger = getLogger(__name__)


def process_image(image, pulse_id, timestamp, x_axis, y_axis, parameters):
    ret = processor.process_image(image, pulse_id, timestamp, x_axis, y_axis, parameters)
    ret["average_value"] = float(ret ["intensity"] ) / len(ret ["x_axis"])/ len(ret ["y_axis"])
    return ret
