from cam_server.pipeline.utils import *
from logging import getLogger
import time
import sys
import os
from collections import deque, OrderedDict
import threading
from threading import Thread

import numpy

from cam_server import config
from cam_server.pipeline.data_processing.pre_processor import process_image as pre_process_image
from cam_server.utils import init_statistics
from cam_server.writer import LAYOUT_DEFAULT, LOCALTIME_DEFAULT, CHANGE_DEFAULT
from cam_server.pipeline.data_processing.functions import is_number, binning, copy_image

_logger = getLogger(__name__)


def run(stop_event, statistics, parameter_queue, cam_client, pipeline_config, output_stream_port,
        background_manager, user_scripts_manager=None):

    camera_name = pipeline_config.get_camera_name()
    set_log_tag(" [" + str(camera_name) + " | " + str(pipeline_config.get_name()) + ":" + str(output_stream_port) + "]")
    exit_code = 0

    def process_bsbuffer(bs_buffer, bs_img_buffer):
        i = 0
        while i < len(bs_buffer):
            bs_pid, bsdata = bs_buffer[i]
            for j in range(len(bs_img_buffer)):
                img_pid = bs_img_buffer[0][0]
                if img_pid < bs_pid:
                    bs_img_buffer.popleft()
                elif img_pid == bs_pid:
                    [pulse_id, [global_timestamp, image, x_axis, y_axis, global_timestamp_float, additional_data]] = bs_img_buffer.popleft()
                    stream_data = OrderedDict()
                    stream_data.update(bsdata)
                    for key, value in bsdata.items():
                        stream_data[key] = value.value
                    if additional_data is not None:
                        try:
                            stream_data.update(additional_data)
                        except:
                            pass
                    process_data(process_image, pulse_id, global_timestamp, image,x_axis, y_axis, global_timestamp_float, stream_data)
                    for k in range(i):
                        bs_buffer.popleft()
                    i = -1
                    break
                else:
                    break
            i = i + 1

    def bs_send_task(bs_buffer, bs_img_buffer, stop_event):
        global sender
        if number_processing_threads <= 0:
            _logger.info("Start bs send thread")
            sender = create_sender(output_stream_port, stop_event)
        try:
            with connect_to_stream() as stream:
                while not stop_event.is_set():
                    message = stream.receive()
                    if not message or stop_event.is_set():
                        if abort_on_timeout():
                            stop_event.set()
                        continue
                    bs_buffer.append([message.data.pulse_id, message.data.data])
                    try:
                        process_bsbuffer(bs_buffer, bs_img_buffer)
                    except Exception as e:
                        _logger.error("Error processing bs buffer: " + str(e))

        except Exception as e:
            _logger.error("Error on bs_send_task: " + str(e))
        finally:
            stop_event.set()
            if sender:
                try:
                    sender.close()
                except:
                    pass
            _logger.info("Exit bs send thread")


    def process_pipeline_parameters():
        parameters = get_pipeline_parameters(pipeline_config, user_scripts_manager)
        _logger.debug("Processing pipeline parameters %s. %s" % (parameters, log_tag))

        background_array = None
        if parameters.get("image_background_enable"):
            background_id = pipeline_config.get_background_id()
            _logger.debug("Image background enabled. Using background_id %s. %s" %(background_id, log_tag))

            try:
                background_array = background_manager.get_background(background_id)
                parameters["image_background_ok"] = True
            except:
                _logger.warning("Invalid background_id: %s. %s" % (background_id, log_tag))
                #if abort_on_error():
                #    raise
                parameters["image_background_ok"] = False
            if background_array is not None:
                background_array = background_array.astype("uint16",copy=False)

        size_x, size_y = cam_client.get_camera_geometry(pipeline_config.get_camera_name())

        by, bx = int(parameters.get("binning_y", 1)), int(parameters.get("binning_x", 1))
        bm = parameters.get("binning_mean", False)
        if (by > 1) or (bx > 1):
            size_x, size_y = int(size_x / bx), int(size_y / by)
            if background_array is not None:
                background_array, _, _ = binning(background_array, None, None, bx, by, bm)
                if background_array.shape != (size_y, size_x):
                    _logger.warning("Bad background shape: %s instead of %s. %s" % (image_background_array.shape, (size_y, size_x), log_tag))

        image_region_of_interest = parameters.get("image_region_of_interest")
        if image_region_of_interest:
            _, size_x, _, size_y = image_region_of_interest

        if size_x and size_y:
            _logger.debug("Image width %d and height %d. %s" % (size_x, size_y, log_tag))


        if parameters.get("rotation"):
            if not isinstance(parameters.get("rotation"), dict):
                parameters["rotation"] = {"angle":float(parameters.get("rotation")), "order":1, "mode":"0.0"}
            if not parameters["rotation"].get("angle"):
               parameters["rotation"] = None
            elif not is_number(parameters["rotation"]["angle"]) or (float(parameters["rotation"]["angle"]) == 0):
                parameters["rotation"] = None
            else:
                if not parameters["rotation"].get("order"):
                    parameters["rotation"]["order"] = 1
                if not parameters["rotation"].get("mode"):
                    parameters["rotation"]["mode"] = "0.0"

        if parameters.get("averaging"):
            try:
                parameters["averaging"] = int(parameters.get("averaging"))
            except:
                parameters["averaging"] = None

        if parameters["mode"] == "FILE":
            if parameters.get("layout") is None:
                parameters["layout"] = LAYOUT_DEFAULT
            if parameters.get("localtime") is None:
                parameters["localtime"] = LOCALTIME_DEFAULT
            if parameters.get("change") is None:
                parameters["change"] = CHANGE_DEFAULT

        if parameters.get("bsread_address"):
            if parameters.get("bsread_image_buf"):
                parameters["bsread_image_buf"] = min(parameters.get("bsread_image_buf"), config.BSREAD_IMAGE_BUFFER_SIZE_MAX)
            else:
                parameters["bsread_image_buf"] =config.BSREAD_IMAGE_BUFFER_SIZE_DEFAULT

            if parameters.get("bsread_data_buf"):
                parameters["bsread_data_buf"] = min(parameters.get("bsread_data_buf"), config.BSREAD_DATA_BUFFER_SIZE_MAX)
            else:
                parameters["bsread_data_buf"] =config.BSREAD_DATA_BUFFER_SIZE_DEFAULT

        return parameters, background_array



    def process_image(pulse_id, global_timestamp, function, image, x_axis, y_axis, global_timestamp_float, bsdata):
        pars = get_parameters()
        try:
            image, x_axis, y_axis = pre_process_image(image, pulse_id, global_timestamp_float, x_axis, y_axis, pars, image_background_array)
            processed_data = function(image, pulse_id, global_timestamp_float, x_axis, y_axis, pars, bsdata)
            #print("Processing PID %d  at proc %d thread %d" % (pulse_id, os.getpid(), threading.get_ident()))
            return processed_data
        except Exception as e:
            _logger.warning("Error processing PID %d at proc %d thread %d: %s" % (pulse_id, os.getpid(), threading.get_ident(), str(e)))
            if abort_on_error():
                raise


    bs_buffer, bs_img_buffer, bs_send_thread = None, None, None

    try:
        init_statistics(statistics)

        pipeline_parameters, image_background_array = process_pipeline_parameters()
        max_frame_rate = pipeline_parameters.get("max_frame_rate")
        averaging = pipeline_parameters.get("averaging")
        rotation = pipeline_parameters.get("rotation")
        copy_images = pipeline_parameters.get("copy")
        if rotation:
            rotation_mode = pipeline_parameters["rotation"]["mode"]
            rotation_angle = int(pipeline_parameters["rotation"]["angle"] / 90) % 4

        connect_to_camera(cam_client)

        _logger.debug("Opening output stream on port %d. %s" % (output_stream_port, log_tag))

        # Indicate that the startup was successful.
        stop_event.clear()

        image_with_stream = has_stream()

        if image_with_stream:
            bs_buffer = deque(maxlen=pipeline_parameters["bsread_data_buf"])
            bs_img_buffer = deque(maxlen=pipeline_parameters["bsread_image_buf"])
            bs_send_thread = Thread(target=bs_send_task, args=(bs_buffer, bs_img_buffer,stop_event))
            bs_send_thread.start()
            if number_processing_threads > 0:
                setup_sender(output_stream_port, stop_event, process_image, user_scripts_manager)
        else:
            setup_sender(output_stream_port, stop_event, process_image, user_scripts_manager)

        _logger.debug("Transceiver started. %s" % (log_tag))
        last_sent_timestamp = 0

        image_buffer = []
        while not stop_event.is_set():
            try:
                while not parameter_queue.empty():
                    new_parameters = parameter_queue.get()
                    pipeline_config.set_configuration(new_parameters)
                    pipeline_parameters, image_background_array = process_pipeline_parameters()
                    max_frame_rate = pipeline_parameters.get("max_frame_rate")
                    averaging = pipeline_parameters.get("averaging")
                    copy_images = pipeline_parameters.get("copy")
                    rotation = pipeline_parameters.get("rotation")
                    if rotation:
                        rotation_mode = pipeline_parameters["rotation"]["mode"]
                        rotation_angle =int(pipeline_parameters["rotation"]["angle"] / 90) % 4

                assert_function_defined()

                pulse_id, global_timestamp, data = receive_stream(True)

                if not data:
                    continue

                image = data["image"].value
                if image is None:
                    continue

                x_axis = data["x_axis"].value
                y_axis = data["y_axis"].value
                if rotation and (rotation_mode == "ortho"):
                    if rotation_angle==1:
                        x_axis,y_axis = y_axis, numpy.flip(x_axis)
                    if rotation_angle == 2:
                        x_axis, y_axis = numpy.flip(x_axis), numpy.flip(y_axis)
                    if rotation_angle == 3:
                        x_axis, y_axis = numpy.flip(y_axis), x_axis
                global_timestamp_float = data["timestamp"].value


                if averaging:
                    continuous = averaging < 0
                    averaging = abs(averaging)
                    if continuous and (len(image_buffer) >= averaging):
                        image_buffer.pop(0)
                    image_buffer.append(image)
                    if (len(image_buffer) >= averaging) or (continuous):
                        try:
                            frames = numpy.array(image_buffer)
                            image = numpy.average(frames, 0)
                        except:
                            #Different shapes
                            image_buffer = []
                            continue
                    else:
                        continue
                else:
                    if copy_images:
                        image = copy_image(image)

                if (not averaging) or (not continuous):
                    image_buffer = []

                #Check maximum frame rate parameter
                if max_frame_rate:
                    min_interval = 1.0 / max_frame_rate
                    if (time.time() - last_sent_timestamp) < min_interval:
                        continue

                additional_data = {}
                if len(data) != len(config.CAMERA_STREAM_REQUIRED_FIELDS):
                    for key, value in data.items():
                        if not key in config.CAMERA_STREAM_REQUIRED_FIELDS:
                                additional_data[key] = value.value

                pars = [global_timestamp, image, x_axis, y_axis, global_timestamp_float, additional_data]
                if image_with_stream:
                    bs_img_buffer.append([pulse_id, pars])
                else:
                    process_data(process_image, pulse_id, *pars)
                last_sent_timestamp = time.time()
            except ProcessingCompleted:
                break
            except Exception as e:
                exit_code = 2
                _logger.exception("Could not process message %s: %s" % (log_tag, str(e)))
                break

    except Exception as e:
        exit_code = 1
        _logger.exception("Exception while trying to start the receive and process thread %s: %s" % (log_tag, str(e)))
        raise

    finally:
        _logger.info("Stopping transceiver. %s" % log_tag)
        stop_event.set()
        cleanup()
        if bs_send_thread:
            try:
                bs_send_thread.join(0.1)
            except:
                pass
        _logger.debug("Exiting process. %s" % log_tag)
        sys.exit(exit_code)
