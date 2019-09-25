import json
from logging import getLogger

import numpy
from cam_server.pipeline.data_processing import functions

_logger = getLogger(__name__)


def process_image(image, pulse_id, timestamp, x_axis, y_axis, parameters):

    # Add return values
    return_value = dict()

    image_threshold = parameters.get("image_threshold")
    if image_threshold is not None and image_threshold > 0:
        functions.apply_threshold(image, image_threshold)

    image_region_of_interest = parameters.get("image_region_of_interest")
    if image_region_of_interest:
        offset_x, size_x, offset_y, size_y = image_region_of_interest
        image = functions.get_region_of_interest(image, offset_x, size_x, offset_y, size_y)

        # Apply roi to geometry x_axis and y_axis
        x_axis = x_axis[offset_x:offset_x + size_x]
        y_axis = y_axis[offset_y:offset_y + size_y]

    # Add return values
    return_value["x_axis"] = x_axis
    return_value["y_axis"] = y_axis
    return_value["image"] = image
    return_value["width"] = image.shape[1]
    return_value["height"] = image.shape[0]
    return_value["timestamp"] = timestamp

    # Needed for config traceability.
    return_value["processing_parameters"] = json.dumps(parameters)
    return return_value
