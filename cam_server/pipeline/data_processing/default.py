from cam_server.pipeline.data_processing.processor import process_image as default_image_process_function
from cam_server.pipeline.data_processing.pre_processor import process_image as pre_process_image


def process_image(image, pulse_id, timestamp, x_axis, y_axis, parameters, image_background_array=None, bsdata = None):
    image, x_axis, y_axis = pre_process_image(image=image,
                         pulse_id=pulse_id,
                         timestamp=timestamp,
                         x_axis=x_axis,
                         y_axis=y_axis,
                         parameters=parameters,
                         image_background_array = image_background_array)

    return default_image_process_function(image=image,
                         pulse_id=pulse_id,
                         timestamp=timestamp,
                         x_axis=x_axis,
                         y_axis=y_axis,
                         parameters=parameters,
                         bsdata=bsdata)
