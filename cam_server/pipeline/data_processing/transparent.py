import json
from logging import getLogger
from cam_server.utils import timestamp_as_float, set_invalid_image
from cam_server.config import PIPELINE_PROCESSING_ERROR

import numpy

_logger = getLogger(__name__)


def process_image(image, pulse_id, timestamp, x_axis, y_axis, parameters, bsdata=None):

    # Add return values
    return_value = dict()

    # Add return values
    return_value["x_axis"] = x_axis
    return_value["y_axis"] = y_axis
    return_value["width"] = image.shape[1]
    return_value["height"] = image.shape[0]
    return_value["timestamp"] =  timestamp_as_float(timestamp)
    return_value["image"] = image
    # Needed for config traceability.
    return_value["processing_parameters"] = json.dumps(parameters)

    if parameters.get(PIPELINE_PROCESSING_ERROR, None):
        return_value["image"] = set_invalid_image(return_value["image"])

    return return_value
