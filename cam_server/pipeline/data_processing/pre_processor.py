import numpy
from logging import getLogger
from cam_server.pipeline.data_processing.functions import rotate, subtract_background, subtract_background_signed, \
    get_region_of_interest, apply_threshold, binning
from cam_server.pipeline.utils import notify_processing_error
_logger = getLogger(__name__)

averaging_buffer = []

def process_image(image, pulse_id, timestamp, x_axis, y_axis, parameters, image_background_array=None):
    global averaging_buffer
    by, bx = int(parameters.get("binning_y", 1)), int(parameters.get("binning_x", 1))
    bm = parameters.get("binning_mean", False)
    if (by > 1) or (bx > 1):
        image, x_axis, y_axis = binning(image, x_axis, y_axis, bx, by, bm)

    if image_background_array is not None:
        if image.shape != image_background_array.shape:
            error = "Bad background image size: %s instead of %s" % (str(image_background_array.shape), str(image.shape))
            _logger.debug("%s - %s" % (error, str(parameters.get("name"))))
            notify_processing_error("Bad background image size")
        else:
            if parameters.get("image_background_enable") == "passive":
                parameters["background_data"] = image_background_array
            elif parameters.get("image_background_enable") == "signed":
                image = subtract_background_signed(image, image_background_array)
            else:
                image = subtract_background(image, image_background_array)

    # Check for rotation parameter
    if parameters.get("mirror_x"):
        image = numpy.fliplr(image)

    if parameters.get("mirror_y"):
        image = numpy.flipud(image)

    rotation = parameters.get("rotation")
    if rotation:
        image = rotate(image, rotation["angle"], rotation["order"], rotation["mode"])

    # Check for ROI
    image_region_of_interest = parameters.get("image_region_of_interest")
    if image_region_of_interest:
        offset_x, size_x, offset_y, size_y = image_region_of_interest
        # Limit ROI to image size
        size_x, size_y = min(size_x, image.shape[1]), min(size_y, image.shape[0])
        offset_x, offset_y = min(offset_x, (image.shape[1] - size_x)), min(offset_y, (image.shape[0] - size_y))
        offset_x, offset_y = max(0, offset_x), max(0, offset_y)

        image = get_region_of_interest(image, offset_x, size_x, offset_y, size_y)

        # Apply roi to geometry x_axis and y_axis
        x_axis = x_axis[offset_x:offset_x + size_x]
        y_axis = y_axis[offset_y:offset_y + size_y]

    # Apply threshold
    image_threshold = parameters.get("image_threshold")
    if image_threshold is not None and image_threshold > 0:
        image = apply_threshold(image, image_threshold)

    #Apply late averaging
    averaging = parameters.get("image_averaging")
    if averaging and (averaging>1):
        while len(averaging_buffer) >= averaging:
            averaging_buffer.pop(0)
        averaging_buffer.append(image)
        try:
            frames = numpy.array(averaging_buffer)
            image = numpy.mean(frames, 0)
            # _logger.info("Averaged: %d" % len(image_buffer))
        except:
            # Different shapes
            image_buffer = []
            return None
    else:
        averaging_buffer = []

    scale = parameters.get("image_scale")
    if scale is not None:
        image = image*scale

    offset = parameters.get("image_offset")
    if offset is not None:
        image = image+offset

    return image, x_axis, y_axis
