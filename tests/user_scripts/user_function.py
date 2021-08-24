from cam_server.pipeline.data_processing import functions, processor

def process_image(image, pulse_id, timestamp, x_axis, y_axis, parameters, bsdata=None):
    ret = processor.process_image(image, pulse_id, timestamp, x_axis, y_axis, parameters, bsdata)
    ret["average_value"] = float(ret ["intensity"] ) / len(ret ["x_axis"])/ len(ret ["y_axis"])
    return ret
