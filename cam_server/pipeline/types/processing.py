from collections import OrderedDict

from cam_server.pipeline.data_processing.functions import is_number, binning, copy_image
from cam_server.pipeline.data_processing.pre_processor import process_image as pre_process_image
from cam_server.pipeline.utils import *
from cam_server.utils import init_statistics, getLogger
from cam_server.writer import LAYOUT_DEFAULT, LOCALTIME_DEFAULT, CHANGE_DEFAULT

_logger = getLogger(__name__)


def run(stop_event, statistics, parameter_queue, logs_queue,cam_client, pipeline_config, output_stream_port,
        background_manager, user_scripts_manager=None):

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
                    [pulse_id, [global_timestamp, image, x_axis, y_axis, additional_data]] = bs_img_buffer.popleft()
                    stream_data = OrderedDict()
                    stream_data.update(bsdata)
                    for key, value in bsdata.items():
                        stream_data[key] = value.value
                    if additional_data is not None:
                        try:
                            stream_data.update(additional_data)
                        except:
                            pass
                    process_data(process_image, pulse_id, global_timestamp, image,x_axis, y_axis, stream_data)
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
            _logger.info("Start bs send thread.%s" % get_log_tag())
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
                        _logger.error("Error processing bs buffer: %s.%s" % (str(e), get_log_tag()))

        except Exception as e:
            _logger.error("Error on bs_send_task: %s.%s" % (str(e), get_log_tag()))
        finally:
            stop_event.set()
            if sender:
                try:
                    sender.close()
                except:
                    pass
            _logger.info("Exit bs send thread.%s" % get_log_tag())


    def process_pipeline_parameters():
        parameters = get_parameters()
        _logger.debug("Processing pipeline parameters %s.%s" % (parameters, get_log_tag()))

        background_array = None
        if parameters.get("image_background_enable"):
            background_id = pipeline_config.get_background_id()
            _logger.debug("Image background enabled. Using background_id %s.%s" %(background_id, get_log_tag()))

            try:
                background_array = background_manager.get_background(background_id)
                parameters["image_background_ok"] = True
            except:
                _logger.warning("Invalid background_id: %s.%s" % (background_id, get_log_tag()))
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
        if background_array is not None:
            if background_array.shape != (size_y, size_x):
                _logger.warning("Bad background shape: %s instead of %s.%s" % (background_array.shape, (size_y, size_x), get_log_tag()))

        image_region_of_interest = parameters.get("image_region_of_interest")
        if image_region_of_interest:
            _, size_x, _, size_y = image_region_of_interest

        if size_x and size_y:
            _logger.debug("Image width %d and height %d.%s" % (size_x, size_y, get_log_tag()))


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



    def process_image(pulse_id, global_timestamp, function, image, x_axis, y_axis, bsdata):
        pars = get_parameters()
        try:
            ret = pre_process_image(image, pulse_id, global_timestamp, x_axis, y_axis, pars, image_background_array)
            if ret is None:
                return
            image, x_axis, y_axis = ret
            processed_data = function(image, pulse_id, global_timestamp, x_axis, y_axis, pars, bsdata)
            #print("Processing PID %d  at proc %d thread %d" % (pulse_id, os.getpid(), threading.get_ident()))
            return processed_data
        except Exception as e:
            _logger.warning("Error processing PID %d at proc %d thread %d: %s.%s" % (pulse_id, os.getpid(), threading.get_ident(), str(e), get_log_tag()))
            if abort_on_error():
                raise

    bs_buffer, bs_img_buffer, bs_send_thread = None, None, None

    try:
        init_statistics(statistics)

        init_pipeline_parameters(pipeline_config, parameter_queue, logs_queue, user_scripts_manager, process_pipeline_parameters, port=output_stream_port)
        pipeline_parameters, image_background_array = process_pipeline_parameters()
        connect_to_camera(cam_client)

        _logger.info("Opening output stream on port %d.%s" % (output_stream_port, get_log_tag()))

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

        _logger.info("Transceiver started.%s" % (get_log_tag()))

        image_buffer = []
        while not stop_event.is_set():
            try:
                ret = check_parameters_changes()
                if ret is not None:
                    pipeline_parameters, image_background_array = ret

                assert_function_defined()

                pulse_id, global_timestamp, data = receive_stream(True)

                if not data:
                    continue

                image = data["image"].value
                if image is None:
                    continue

                x_axis = data["x_axis"].value
                y_axis = data["y_axis"].value

                if pipeline_parameters.get("mirror_x"):
                    x_axis = numpy.flip(x_axis)

                if pipeline_parameters.get("mirror_y"):
                    y_axis = numpy.flip(y_axis)

                if pipeline_parameters.get("rotation"):
                    if pipeline_parameters["rotation"]["mode"] == "ortho":
                        rotation_angle = int(pipeline_parameters["rotation"]["angle"] / 90) % 4
                        if rotation_angle == 1:
                            x_axis,y_axis = y_axis, numpy.flip(x_axis)
                        if rotation_angle == 2:
                            x_axis, y_axis = numpy.flip(x_axis), numpy.flip(y_axis)
                        if rotation_angle == 3:
                            x_axis, y_axis = numpy.flip(y_axis), x_axis

                averaging = pipeline_parameters.get("averaging")
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
                    if pipeline_parameters.get("copy"):
                        image = copy_image(image)

                if (not averaging) or (not continuous):
                    image_buffer = []

                additional_data = {}
                if len(data) != len(config.CAMERA_STREAM_REQUIRED_FIELDS):
                    for key, value in data.items():
                        if not key in config.CAMERA_STREAM_REQUIRED_FIELDS:
                                additional_data[key] = value.value

                pars = [global_timestamp, image, x_axis, y_axis, additional_data]
                if image_with_stream:
                    bs_img_buffer.append([pulse_id, pars])
                else:
                    process_data(process_image, pulse_id, *pars)
            except ProcessingCompleted:
                break
            except Exception as e:
                exit_code = 2
                _logger.exception("Error in pipeline processing: %s.%s" % (str(e), get_log_tag()))
                break

    except Exception as e:
        exit_code = 1
        _logger.exception("Exception trying to start the receive thread: %s.%s" % (str(e), get_log_tag()))
        raise

    finally:
        _logger.info("Stopping transceiver.%s" % get_log_tag())
        stop_event.set()
        if bs_send_thread:
            try:
                bs_send_thread.join(0.1)
            except:
                pass
        cleanup(exit_code)
