import json
from logging import getLogger
from cam_server.utils import timestamp_as_float

import numpy

_logger = getLogger(__name__)


def process_image(image, pulse_id, timestamp, x_axis, y_axis, parameters, bsdata=None):

    # Add return values
    return_value = dict()

    # Add return values
    return_value["x_axis"] = x_axis
    return_value["y_axis"] = y_axis
    return_value["image"] = image
    return_value["width"] = image.shape[1]
    return_value["height"] = image.shape[0]
    return_value["timestamp"] =  timestamp_as_float(timestamp)

    # Needed for config traceability.
    return_value["processing_parameters"] = json.dumps(parameters)
    return return_value
