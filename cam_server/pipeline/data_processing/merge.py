
from cam_server.pipeline.data_processing import functions, processor


def process_image(image, pulse_id, timestamp, x_axis, y_axis, parameters, bsdata):
    ret = processor.process_image(image, pulse_id, timestamp, x_axis, y_axis, parameters, bsdata)
    ret.update(bsdata)
    return ret
