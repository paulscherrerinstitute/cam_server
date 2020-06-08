import numpy
from cam_server.pipeline.data_processing.functions import chunk_copy


def transform_image(image, camera_config):

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
        image = chunk_copy(image)
        mask_for_zeros = (background > image)
        numpy.subtract(image, background, image)
        image[mask_for_zeros] = 0


    if not image.flags['C_CONTIGUOUS']:
        image = numpy.ascontiguousarray(image)

    return image
