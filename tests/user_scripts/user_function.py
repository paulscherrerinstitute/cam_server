import time
from cam_server.pipeline.data_processing import functions, processor

count = 10
def process_image(image, pulse_id, timestamp, x_axis, y_axis, parameters, bsdata=None):
    global count
    count = count + 1
    ret = processor.process_image(image, pulse_id, timestamp, x_axis, y_axis, parameters, bsdata)
    ret["average_value"] = float(ret ["intensity"] ) / len(ret ["x_axis"])/ len(ret ["y_axis"])
    if count % 10 == 0:
         ret["average_value"] = int(ret["average_value"])
    return ret
