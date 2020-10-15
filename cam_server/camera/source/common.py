from logging import getLogger
import numpy
from cam_server.pipeline.data_processing.functions import chunk_copy, binning

_logger = getLogger(__name__)

def transform_image(image, camera_config):
    by, bx = int(camera_config.parameters.get("binning_y", 1)), int(camera_config.parameters.get("binning_x", 1))
    bm = camera_config.parameters.get("binning_mean", False)
    if (by > 1) or (bx > 1):
        image, _, _ = binning(image, None, None, bx, by, bm)

    if camera_config.parameters["mirror_x"]:
        image = numpy.fliplr(image)

    if camera_config.parameters["mirror_y"]:
        image = numpy.flipud(image)

    if camera_config.parameters["rotate"] != 0:
        image = numpy.rot90(image, camera_config.parameters["rotate"])

    if camera_config.parameters["roi"]:
        offset_x, size_x, offset_y, size_y = camera_config.parameters["roi"]
        image = image[offset_y:offset_y + size_y, offset_x:offset_x + size_x]

    background = camera_config.parameters.get("background_data")
    if background is not None:
        if (by > 1) or (bx > 1):
            background, _, _ = binning(background, None, None, bx, by, bm)
        if background.shape == image.shape:
            image = chunk_copy(image)
            mask_for_zeros = (background > image)
            numpy.subtract(image, background, image)
            image[mask_for_zeros] = 0
        else:
            _logger.info("Bad background shape for camera %s: %s instead of %s" % (camera_config.get_source(),background.shape, image.shape))

    if not image.flags['C_CONTIGUOUS']:
        image = numpy.ascontiguousarray(image)

    return image
