from logging import getLogger
from importlib import import_module
from imp import load_source
import time
import sys
import os
from collections import deque, OrderedDict
from threading import Thread, Event, RLock
import multiprocessing

import numpy
import json

from bsread import Source, PUB, SUB, PUSH, PULL, DEFAULT_DISPATCHER_URL
from bsread import source as bssource
from bsread.sender import Sender

from cam_server import config
from cam_server.pipeline.data_processing.processor import process_image as default_image_process_function
from cam_server.pipeline.data_processing.pre_processor import process_image as pre_process_image
from cam_server.utils import get_host_port_from_stream_address, set_statistics, on_message_sent, init_statistics, MaxLenDict, get_clients
from cam_server.writer import WriterSender, UNDEFINED_NUMBER_OF_RECORDS, LAYOUT_DEFAULT, LOCALTIME_DEFAULT, CHANGE_DEFAULT
from cam_server.pipeline.data_processing.functions import chunk_copy, is_number, binning, copy_image

from cam_server.ipc import IpcSource

_logger = getLogger(__name__)

class ProcessingCompleated(Exception):
     pass


def init_sender(sender, pipeline_parameters):
    sender.record_count = 0
    sender.data_format = None
    create_header = pipeline_parameters.get("create_header")
    if create_header in (True,"always"):
        sender.create_header = True
    elif create_header in (False, "once"):
        sender.create_header = False
    else:
        sender.create_header = None
    sender.records=pipeline_parameters.get("records")

def create_sender(pipeline_parameters, output_stream_port, stop_event, log_tag):
    sender = None
    def no_client_action():
        nonlocal sender,pipeline_parameters
        if pipeline_parameters["no_client_timeout"] > 0:
            _logger.warning("No client connected to the pipeline stream for %d seconds. Closing instance. %s",
                            pipeline_parameters["no_client_timeout"], log_tag)
            stop_event.set()
            if sender:
                if pipeline_parameters["mode"] == "PUSH" and pipeline_parameters["block"]:
                    _logger.warning("Killing the process: cannot stop gracefully if sender is blocking")
                    os._exit(0)

    if pipeline_parameters["mode"] == "FILE":
        file_name = pipeline_parameters["file"]
        records = pipeline_parameters.get("records")
        sender = WriterSender(output_file=file_name,
                              number_of_records=records if records else UNDEFINED_NUMBER_OF_RECORDS,
                              layout=pipeline_parameters["layout"],
                              save_local_timestamps=pipeline_parameters["localtime"],
                              change=pipeline_parameters["change"],
                              attributes={})
    else:
        sender = Sender(port=output_stream_port,
                        mode=PUSH if (pipeline_parameters["mode"] == "PUSH") else PUB,
                        queue_size=pipeline_parameters["queue_size"],
                        block=pipeline_parameters["block"],
                        data_header_compression=config.CAMERA_BSREAD_DATA_HEADER_COMPRESSION)
    sender.open(no_client_action=no_client_action, no_client_timeout=pipeline_parameters["no_client_timeout"]
                if pipeline_parameters["no_client_timeout"] > 0 else sys.maxsize)
    init_sender(sender, pipeline_parameters)
    sender.last_client_update = time.time()
    return sender


def check_records(sender, pipeline_parameters):
    records = pipeline_parameters.get("records")
    if records:
        sender.record_count = sender.record_count + 1
        if sender.record_count >= records:
            raise ProcessingCompleated("Reached number of records: " + str(records))

def send(sender, data, timestamp, pulse_id, pipeline_parameters, statistics):
    try:
        if sender.create_header == True:
            check_header = True
        elif sender.create_header == False:
            check_header = (sender.data_format is None)
            sender.data_format = True
        else:
            #data_format = {k: ((v.shape, v.dtype) if isinstance(v, numpy.ndarray) else type(v)) for k, v in data.items()}
            data_format = {k: ((v.shape, v.dtype) if isinstance(v, numpy.ndarray) else
                            (len(v) if isinstance(v, list) else type(v))) for k, v in data.items()}
            check_header = data_format != sender.data_format
            if check_header:
                sender.data_format = data_format
        sender.send(data=data, timestamp=timestamp, pulse_id=pulse_id, check_data=check_header)
        on_message_sent(statistics)
        if sender.records:
            check_records(sender, pipeline_parameters)
    except Exception as e :
        raise


def get_pipeline_parameters(pipeline_config):
    parameters = pipeline_config.get_configuration()
    if parameters.get("no_client_timeout") is None:
        parameters["no_client_timeout"] = config.MFLOW_NO_CLIENTS_TIMEOUT

    if parameters.get("queue_size") is None:
        parameters["queue_size"] = config.PIPELINE_DEFAULT_QUEUE_SIZE

    if parameters.get("mode") is None:
        parameters["mode"] = config.PIPELINE_DEFAULT_MODE

    if parameters.get("block") is None:
        parameters["block"] = config.PIPELINE_DEFAULT_BLOCK

    if parameters.get("pid_range"):
        try:
            parameters["pid_range"] = int(parameters.get("pid_range")[0]), int(parameters.get("pid_range")[1])
        except:
            parameters["pid_range"] = None
    return parameters

def get_dispatcher_parameters(parameters):
    dispatcher_url = parameters.get("dispatcher_url")
    if dispatcher_url is None:
        dispatcher_url = DEFAULT_DISPATCHER_URL
    dispatcher_verify_request = parameters.get("dispatcher_verify_request")
    if dispatcher_verify_request is None:
        dispatcher_verify_request = True
    dispatcher_disable_compression = parameters.get("dispatcher_disable_compression")
    if dispatcher_disable_compression is None:
        dispatcher_disable_compression = False
    return dispatcher_url, dispatcher_verify_request, dispatcher_disable_compression

functions = {}

def get_function(pipeline_parameters, user_scripts_manager, log_tag):
    name = pipeline_parameters.get("function")
    if not name:
        if pipeline_parameters.get("pipeline_type") == config.PIPELINE_TYPE_STREAM:
            return None
        return default_image_process_function
    try:
        f = functions.get(name)
        reload = pipeline_parameters.get("reload")
        if (not f) or reload:
            if (not f):
                _logger.info("Importing function: %s. %s" % (name, log_tag))
            else:
                _logger.info("Reloading function: %s. %s" % (name, log_tag))
            if '/' in name:
                mod = load_source('mod', name)
            else:
                if user_scripts_manager and user_scripts_manager.exists(name):
                    mod = load_source('mod', user_scripts_manager.get_path(name))
                else:
                    mod = import_module("cam_server.pipeline.data_processing." + str(name))
            try:
                functions[name] = f = mod.process_image
            except:
                functions[name] = f = mod.process
            pipeline_parameters["reload"] = False
        return f
    except:
        _logger.exception("Could not import function: %s. %s" % (str(name), log_tag))
        return None

def create_source(camera_stream_address, receive_timeout=config.PIPELINE_RECEIVE_TIMEOUT, mode=SUB):
    source_host, source_port = get_host_port_from_stream_address(camera_stream_address)
    if camera_stream_address.startswith("ipc"):
        return IpcSource(address=camera_stream_address,receive_timeout=receive_timeout, mode=mode)
    else:
        return Source(host=source_host, port=source_port, receive_timeout=receive_timeout, mode=mode)


def resolve_camera_source(cam_client, pipeline_config):
    source_mode = SUB
    if pipeline_config.get_input_stream():
        camera_stream_address = pipeline_config.get_input_stream()
        source_mode = pipeline_config.get_input_mode()
    else:
        camera_stream_address = cam_client.get_instance_stream(pipeline_config.get_camera_name())
    source_host, source_port = get_host_port_from_stream_address(camera_stream_address)
    return camera_stream_address, source_host, source_port, source_mode


def processing_pipeline(stop_event, statistics, parameter_queue,
                        cam_client, pipeline_config, output_stream_port, background_manager, user_scripts_manager = None):
    camera_name = pipeline_config.get_camera_name()
    log_tag = " [" + str(camera_name) + " | " + str(pipeline_config.get_name()) + ":" + str(output_stream_port) + "]"
    source = None
    camera_host = None
    camera_port = None
    sender = None
    number_processing_threads = 0
    multiprocessed = False
    processing_thread_index = 0
    exit_code = 0


    def connect_to_camera():
        nonlocal source, camera_host, camera_port
        camera_stream_address, source_host, source_port, source_mode = resolve_camera_source(cam_client, pipeline_config)
        _logger.warning("Connecting to camera stream address %s. %s" % (camera_stream_address, log_tag))
        if source is None or source_host != camera_host or source_port != camera_port:
            if source:
                try:
                    source.disconnect()
                    source = None
                except:
                    pass
            source = create_source(camera_stream_address, mode=source_mode)
            source.connect()
            camera_host, camera_port = source_host, source_port

    def message_buffer_send_task(message_buffer, stop_event):
        nonlocal sender
        _logger.info("Start message buffer send thread")
        sender = create_sender(pipeline_parameters, output_stream_port, stop_event, log_tag)
        try:
            while not stop_event.is_set():
                if len(message_buffer) == 0:
                    time.sleep(0.01)
                else:
                    (processed_data, timestamp, pulse_id) = message_buffer.popleft()
                    send(sender, processed_data, timestamp, pulse_id, pipeline_parameters, statistics)

        except Exception as e:
            _logger.error("Error on message buffer send thread" + str(e))
        finally:
            stop_event.set()
            if sender:
                try:
                    sender.close()
                except:
                    pass
            _logger.info("Exit message buffer send thread")

    def process_bsbuffer(bs_buffer, bs_img_buffer, sender):
        i = 0
        while i < len(bs_buffer):
            bs_pid, bsdata = bs_buffer[i]
            for j in range(len(bs_img_buffer)):
                img_pid = bs_img_buffer[0][0]
                if img_pid < bs_pid:
                    bs_img_buffer.popleft()
                elif img_pid == bs_pid:
                    [pulse_id, [function, global_timestamp, global_timestamp_float, image, pulse_id, x_axis, y_axis, pipeline_parameters]] = bs_img_buffer.popleft()
                    stream_data = OrderedDict()
                    for key, value in bsdata.items():
                        stream_data[key] = value.value
                    on_receive_data(function, global_timestamp, global_timestamp_float, sender, None, image, pulse_id, x_axis, y_axis, pipeline_parameters, stream_data)
                    for k in range(i):
                        bs_buffer.popleft()
                    i = -1
                    break
                else:
                    break
            i = i + 1

    def bs_send_task(bs_buffer, bs_img_buffer, bsread_address, bsread_channels, bsread_mode, dispatcher_parameters, stop_event):
        dispatcher_url, dispatcher_verify_request, dispatcher_disable_compression = dispatcher_parameters
        nonlocal sender
        if bsread_address:
            bsread_host, bsread_port = get_host_port_from_stream_address(bsread_address)
            bsread_mode = SUB if bsread_mode == "SUB" else PULL
        else:
            bsread_host, bsread_port =None, 9999
            bsread_mode = PULL if bsread_mode == "PULL" else SUB


        _logger.info("Start bs send thread")
        sender = create_sender(pipeline_parameters, output_stream_port, stop_event, log_tag)

        try:
            with bssource(host=bsread_host,
                          port=bsread_port,
                          mode=bsread_mode,
                          channels=bsread_channels,
                          receive_timeout=config.PIPELINE_RECEIVE_TIMEOUT,
                          dispatcher_url=dispatcher_url,
                          dispatcher_verify_request=dispatcher_verify_request,
                          dispatcher_disable_compression=dispatcher_disable_compression
                          ) as stream:
                while not stop_event.is_set():
                    message = stream.receive()
                    if not message or stop_event.is_set():
                        continue
                    bs_buffer.append([message.data.pulse_id, message.data.data])
                    try:
                        process_bsbuffer(bs_buffer, bs_img_buffer, sender)
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
        parameters = get_pipeline_parameters(pipeline_config)
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
                #if parameters.get("abort_on_error", config.ABORT_ON_ERROR):
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

        if not parameters.get("camera_timeout"):
            parameters["camera_timeout"] = 10.0

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

    def process_thread_task(thread_buffer, tx_buffer, tx_lock, stop_event, index):
        _logger.info("Start processing thread %d" % index)
        try:
            while not stop_event.is_set():
                if len(thread_buffer) == 0:
                    time.sleep(0.01)
                else:
                    (function, global_timestamp, global_timestamp_float, sender, message_buffer, image, pulse_id, x_axis, y_axis, parameters, bsdata) = thread_buffer.popleft()

                    processed_data = process_image(function, image, x_axis, y_axis, pulse_id, global_timestamp_float, bsdata, thread_index=index)
                    if processed_data is None:
                        with tx_lock:
                            try:
                                received_pids.remove(pulse_id)
                            except:
                                _logger.warning("Error removing PID %d at %d" % (pulse_id, index))
                    else:
                        with tx_lock:
                            tx_buffer[pulse_id]= (processed_data, global_timestamp, pulse_id, message_buffer)

        except Exception as e:
            _logger.error("Error on processing thread %d: %s" % (index, str(e)))
        finally:
            stop_event.set()
            _logger.info("Exit processing thread %d" % index)

    def threaded_processing_send_task(tx_buffer, tx_lock, received_pids, stop_event):
        nonlocal sender
        _logger.info("Start threaded processing send thread")
        last_sent = -1
        sender = create_sender(pipeline_parameters, output_stream_port, stop_event, log_tag)
        try:
            while not stop_event.is_set():
                tx = None
                popped = False
                with tx_lock:
                    size = len(tx_buffer)
                    if (size > 0) and (len(received_pids) > 0):
                        pid = received_pids[0]
                        if pid in tx_buffer.keys():
                            received_pids.popleft()
                            tx = (processed_data, global_timestamp, pulse_id, message_buffer) = tx_buffer.pop(pid)
                        else:
                            if size >= tx_buffer.maxlen:
                                received_pids.popleft()
                                _logger.error("Failed processing Pulse ID" + str(pulse_id))
                                popped = True
                if tx is not None:
                    send_data(sender, processed_data, global_timestamp, pulse_id, message_buffer)

                    #_logger.info("Sent PID %d" % (pulse_id))
                    #if last_sent != -1:
                    #    if pulse_id != (last_sent + 1):
                    #        print("Error TX: expected %d - sent %d" %(last_sent + 1, pulse_id))
                    last_sent = pulse_id

                elif not popped:
                    time.sleep(0.001)

        except Exception as e:
            _logger.error("Error on threaded processing send thread" + str(e))
        finally:
            stop_event.set()
            if sender:
                try:
                    sender.close()
                except:
                    pass
            _logger.info("Exit threaded processing send thread")

    def process_task(thread_buffer, tx_queue, stop_event, index, user_scripts_manager, log_tag):
        _logger.info("Start process %d" % index)

        try:
            while not stop_event.is_set():
                try:
                    msg = thread_buffer.get(False)

                except:
                    time.sleep(0.001)
                    continue

                (global_timestamp, global_timestamp_float, message_buffer, image, pulse_id, x_axis, y_axis, parameters, bsdata) = msg

                function = get_function(parameters, user_scripts_manager, log_tag)
                if function is not None:
                    processed_data = process_image(function, image, x_axis, y_axis, pulse_id, global_timestamp_float, bsdata, thread_index=index)
                    try:
                        tx_queue.put((processed_data, global_timestamp, pulse_id, message_buffer), False)
                    except Exception as e:
                        _logger.error("Error adding to tx buffer %d: %s" % (index, str(e)))
                    #_logger.info("Processed message %d in process %d" % (pulse_id, index))


        except Exception as e:
            _logger.error("Error on process %d: %s" % (index, str(e)))
            import traceback
            traceback.print_exc()
        finally:
            stop_event.set()
            _logger.info("Exit process  %d" % index)


    def process_send_task(tx_queue, received_pids_queue, spawn_send_thread, stop_event):
        sender = None
        _logger.info("Start send process")
        received_pids = deque()
        tx_buffer = MaxLenDict(maxlen=(thread_buffer_size * number_processing_threads))


        last_sent = -1

        if spawn_send_thread:
            tx_buffer_lock = RLock()
            send_thread = Thread(target=threaded_processing_send_task,
                                            args=(tx_buffer, tx_buffer_lock, received_pids, stop_event))
            send_thread.start()
        else:
            sender = create_sender(pipeline_parameters, output_stream_port, stop_event, log_tag)

        try:
            while not stop_event.is_set():
                try:
                    tx = (processed_data, global_timestamp, pulse_id, message_buffer) = tx_queue.get(False)
                except:
                    time.sleep(0.001)
                    continue


                while True:
                    try:
                        received_pids.append(received_pids_queue.get(False))
                    except:
                        break
                if tx is not None:
                    if spawn_send_thread:
                        with tx_buffer_lock:
                            tx_buffer[pulse_id] = tx
                    else:
                        tx_buffer[pulse_id] = tx
                        while len(received_pids)>0:
                            tx = None
                            size = len(tx_buffer)
                            pid = received_pids[0]
                            if pid in tx_buffer.keys():
                                received_pids.popleft()
                                tx = (processed_data, global_timestamp, pulse_id, message_buffer) = tx_buffer.pop(pid)
                            else:
                                if size >= tx_buffer.maxlen:
                                    received_pids.popleft()
                                    _logger.error("Failed processing Pulse ID" + str(pulse_id))
                            if tx is not None:
                                send_data(sender, processed_data, global_timestamp, pulse_id, message_buffer)
                                #if last_sent!=-1:
                                #    if pulse_id != (last_sent+1):
                                #        print ("Error TX: expected %d - sent %d", last_sent+1, pulse_id)
                                last_sent = pulse_id
                            else:
                                break

        except Exception as e:
            _logger.error("Error send process" + str(e))
            import traceback
            traceback.print_exc()
        finally:
            stop_event.set()
            if sender:
                try:
                    sender.close()
                except:
                    pass
            _logger.info("Exit send process")


    def send_data(sender, processed_data, global_timestamp, pulse_id, message_buffer = None):
        nonlocal last_sent_timestamp
        if processed_data is not None:
            # Requesting subset of the data
            include = pipeline_parameters.get("include")
            if include:
                aux = {}
                for key in include:
                    aux[key] = processed_data.get(key)
                processed_data = aux
            exclude = pipeline_parameters.get("exclude")
            if exclude:
                for field in exclude:
                    processed_data.pop(field, None)

            last_sent_timestamp = time.time()
            if message_buffer:
                message_buffer.append((processed_data, global_timestamp, pulse_id))
            else:
                send(sender, processed_data, global_timestamp, pulse_id, pipeline_parameters, statistics)
            # When multiprocessed cannot access sender object from main process
            if multiprocessed and ((time.time() - sender.last_client_update) > 1.0):
                statistics.num_clients = get_clients(sender)
                sender.last_client_update = time.time()

            #_logger.debug("Sent PID %d" % (pulse_id,))

    def process_image(function, image, x_axis, y_axis, pulse_id, global_timestamp_float, bsdata, thread_index=0):
        try:
            image, x_axis, y_axis = pre_process_image(image, pulse_id, global_timestamp_float, x_axis, y_axis, pipeline_parameters, image_background_array)
            processed_data = function(image, pulse_id, global_timestamp_float, x_axis, y_axis, pipeline_parameters, bsdata)
            _logger.debug("Processed PID %d at thread %d" % (pulse_id, thread_index))
            return processed_data
        except Exception as e:
            _logger.warning("Error processing PID %d at thread %d: %s" % (pulse_id, thread_index, str(e)))
            if pipeline_parameters.get("abort_on_error", config.ABORT_ON_ERROR):
                raise

    def on_receive_data(function, global_timestamp, global_timestamp_float, sender, message_buffer, image, pulse_id, x_axis, y_axis, parameters, bsdata=None):
        nonlocal number_processing_threads, multiprocessed , processing_thread_index, received_pids

        if number_processing_threads > 0:
            thread_buffer = thread_buffers[processing_thread_index]
            processing_thread_index = processing_thread_index + 1
            if processing_thread_index >= number_processing_threads:
                processing_thread_index = 0
            if multiprocessed:
                received_pids.put(pulse_id)
                thread_buffer.put((global_timestamp, global_timestamp_float, message_buffer, image, pulse_id, x_axis,
                                  y_axis, parameters, bsdata))
            else:
                #Deque append and popleft are supposed to be thread safe
                thread_buffer.append((function, global_timestamp, global_timestamp_float, sender, message_buffer, image, pulse_id, x_axis, y_axis, parameters, bsdata))
                with tx_lock:
                    received_pids.append(pulse_id)
            return
        processed_data = process_image(function, image, x_axis, y_axis, pulse_id, global_timestamp_float, bsdata)
        if processed_data is not None:
            send_data(sender, processed_data, global_timestamp, pulse_id, message_buffer)


    source, sender = None, None
    message_buffer, message_buffer_send_thread  = None, None
    bs_buffer, bs_img_buffer, bs_send_thread = None, None, None
    processing_threads = []

    try:
        init_statistics(statistics)

        pipeline_parameters, image_background_array = process_pipeline_parameters()
        camera_timeout = pipeline_parameters.get("camera_timeout")
        debug = pipeline_parameters.get("debug")
        pause = pipeline_parameters.get("pause")
        pid_range = pipeline_parameters.get("pid_range")
        downsampling = pipeline_parameters.get("downsampling")
        max_frame_rate = pipeline_parameters.get("max_frame_rate")
        averaging = pipeline_parameters.get("averaging")
        rotation = pipeline_parameters.get("rotation")
        copy_images = pipeline_parameters.get("copy")
        if rotation:
            rotation_mode = pipeline_parameters["rotation"]["mode"]
            rotation_angle = int(pipeline_parameters["rotation"]["angle"] / 90) % 4

        current_pid, former_pid = None, None
        connect_to_camera()

        _logger.debug("Opening output stream on port %d. %s" % (output_stream_port, log_tag))

        number_processing_threads = pipeline_parameters.get("processing_threads", 0)
        thread_buffers = None if number_processing_threads==0 else []
        multiprocessed = pipeline_parameters.get("multiprocessing", False)
        bsread_address = pipeline_parameters.get("bsread_address")
        bsread_channels = pipeline_parameters.get("bsread_channels")
        bsread_mode = pipeline_parameters.get("bsread_mode")
        dispatcher_parameters = get_dispatcher_parameters(pipeline_parameters)

        if bsread_channels is not None:
            if type(bsread_channels) != list:
                bsread_channels = json.loads(bsread_channels)
            if len(bsread_channels) == 0:
                bsread_channels = None

        # Indicate that the startup was successful.
        stop_event.clear()

        image_with_stream = bsread_address or (bsread_channels is not None)
        if image_with_stream:
            bs_buffer = deque(maxlen=pipeline_parameters["bsread_data_buf"] )
            bs_img_buffer = deque(maxlen=pipeline_parameters["bsread_image_buf"] )
            bs_send_thread = Thread(target=bs_send_task, args=(bs_buffer, bs_img_buffer, bsread_address, bsread_channels, bsread_mode, dispatcher_parameters, stop_event))
            bs_send_thread.start()

        else:
            buffer_size = pipeline_parameters.get("buffer_size")
            if buffer_size:
                message_buffer = deque(maxlen=buffer_size)
                message_buffer_send_thread = Thread(target=message_buffer_send_task, args=(message_buffer, stop_event))
                message_buffer_send_thread.start()
            elif number_processing_threads > 0:
                processing_thread_index = 0
                thread_buffer_size = pipeline_parameters.get("thread_buffer_size", 10)
                if multiprocessed:
                    tx_lock = multiprocessing.Lock()

                    if pipeline_parameters.get("processing_manager", False):
                        received_pids = multiprocessing.Manager().Queue()
                        tx_queue = multiprocessing.Manager().Queue(thread_buffer_size * number_processing_threads)
                    else:
                        received_pids = multiprocessing.Queue()
                        tx_queue = multiprocessing.Queue(thread_buffer_size * number_processing_threads)
                    spawn_send_thread = pipeline_parameters.get("spawn_send_thread", True)
                    message_buffer_send_thread= multiprocessing.Process(target=process_send_task,args=(tx_queue, received_pids, spawn_send_thread, stop_event))
                    message_buffer_send_thread.start()
                    for i in range(number_processing_threads):
                        thread_buffer = multiprocessing.Queue(thread_buffer_size)
                        thread_buffers.append(thread_buffer)
                        processing_thread = multiprocessing.Process(target=process_task, args=(thread_buffer, tx_queue, stop_event, i, user_scripts_manager, log_tag))
                        processing_threads.append(processing_thread)
                        processing_thread.start()
                else:
                    tx_lock = RLock()
                    received_pids = deque()
                    tx_buffer = MaxLenDict(maxlen=(thread_buffer_size * number_processing_threads))
                    message_buffer_send_thread = Thread(target=threaded_processing_send_task, args=(tx_buffer, tx_lock, received_pids, stop_event))
                    message_buffer_send_thread.start()
                    for i in range(number_processing_threads):
                        thread_buffer = deque(maxlen=thread_buffer_size)
                        thread_buffers.append(thread_buffer)
                        processing_thread = Thread(target=process_thread_task, args=(thread_buffer, tx_buffer, tx_lock, stop_event, i))
                        processing_threads.append(processing_thread)
                        processing_thread.start()
            else:
                sender = create_sender(pipeline_parameters, output_stream_port, stop_event, log_tag)

        function = get_function(pipeline_parameters, user_scripts_manager, log_tag)
        eval_function = not multiprocessed or (number_processing_threads <= 0) or (image_with_stream)

        _logger.debug("Transceiver started. %s" % (log_tag))
        downsampling_counter = sys.maxsize  # The first is always sent
        last_sent_timestamp = 0
        last_rcvd_timestamp = time.time()

        image_buffer = []
        while not stop_event.is_set():
            try:
                while not parameter_queue.empty():
                    new_parameters = parameter_queue.get()
                    pipeline_config.set_configuration(new_parameters)
                    pipeline_parameters, image_background_array = process_pipeline_parameters()
                    camera_timeout = pipeline_parameters.get("camera_timeout")
                    debug = pipeline_parameters.get("debug")
                    pause = pipeline_parameters.get("pause")
                    pid_range = pipeline_parameters.get("pid_range")
                    downsampling = pipeline_parameters.get("downsampling")
                    max_frame_rate = pipeline_parameters.get("max_frame_rate")
                    averaging = pipeline_parameters.get("averaging")
                    copy_images = pipeline_parameters.get("copy")
                    rotation = pipeline_parameters.get("rotation")
                    if rotation:
                        rotation_mode = pipeline_parameters["rotation"]["mode"]
                        rotation_angle =int(pipeline_parameters["rotation"]["angle"] / 90) % 4
                    function = get_function(pipeline_parameters, user_scripts_manager, log_tag)

                frame_shape = None
                data = source.receive()
                if data:
                    image = data.data.data["image"].value
                    if image is not None:
                        frame_shape = str(image.shape[1]) + "x" + str(image.shape[0]) + "x" + str(image.itemsize)
                    last_rcvd_timestamp = time.time()
                set_statistics(statistics, sender, data.statistics.total_bytes_received if data else statistics.total_bytes,  1 if data else 0, frame_shape)

                if not data:
                    if camera_timeout:
                        if (camera_timeout > 0) and (time.time() - last_rcvd_timestamp) > camera_timeout:
                            _logger.warning("Camera timeout. %s" % log_tag)
                            current_pid, former_pid = None, None
                            last_rcvd_timestamp = time.time()
                            #Try reconnecting to the camera. If fails raise exception and stops pipeline.
                            connect_to_camera()
                    continue

                pulse_id = data.data.pulse_id
                if debug:
                    if (former_pid is not None) and (current_pid is not None):
                        if pulse_id != (current_pid + (current_pid - former_pid)):
                            _logger.warning("Unexpected PID: " + str(pulse_id) + " -  previous: " + str(former_pid) + ", " + str(current_pid) )
                            current_pid, former_pid = None, None
                former_pid = current_pid
                current_pid = pulse_id

                if pause:
                    continue

                if pid_range:
                    if (pid_range[0]<=0) or (pulse_id < pid_range[0]):
                        continue
                    elif (pid_range[1]>0) and (pulse_id > pid_range[1]):
                        _logger.warning("Reached end of pid range: stopping pipeline")
                        raise ProcessingCompleated("End of pid range")

                # Check downsampling parameter
                if downsampling:
                    downsampling_counter += 1
                    if downsampling_counter > downsampling:
                        downsampling_counter = 0
                    else:
                        continue
                #Check maximum frame rate parameter
                if max_frame_rate:
                    min_interval = 1.0 / max_frame_rate
                    if (time.time() - last_sent_timestamp) < min_interval:
                        continue

                if image is None:
                    continue

                x_axis = data.data.data["x_axis"].value
                y_axis = data.data.data["y_axis"].value
                if rotation and (rotation_mode == "ortho"):
                    if rotation_angle==1:
                        x_axis,y_axis = y_axis, numpy.flip(x_axis)
                    if rotation_angle == 2:
                        x_axis, y_axis = numpy.flip(x_axis), numpy.flip(y_axis)
                    if rotation_angle == 3:
                        x_axis, y_axis = numpy.flip(y_axis), x_axis

                global_timestamp = (data.data.global_timestamp, data.data.global_timestamp_offset)
                global_timestamp_float = data.data.data["timestamp"].value

                if function is None:
                    return

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

                # image, x_axis, y_axis = pre_process_image(image, x_axis, y_axis, image_background_array, pipeline_parameters)
                if image_with_stream:
                    bs_img_buffer.append([pulse_id,
                                          [function, global_timestamp, global_timestamp_float, image, pulse_id, x_axis,
                                           y_axis,
                                           pipeline_parameters]])
                else:
                    additional_data = {}
                    if len(data.data.data) != len(config.CAMERA_STREAM_REQUIRED_FIELDS):
                        for key, value in data.data.data.items():
                            if not key in config.CAMERA_STREAM_REQUIRED_FIELDS:
                                additional_data[key] = value.value
                    on_receive_data(function, global_timestamp, global_timestamp_float, sender, message_buffer, image,
                                 pulse_id, x_axis, y_axis, pipeline_parameters, additional_data)
            except ProcessingCompleated:
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

        if source:
            try:
                source.disconnect()
            except:
                pass

        if message_buffer_send_thread:
            try:
                message_buffer_send_thread.join(0.1)
            except:
                pass
        else:
            if sender:
                try:
                    sender.close()
                except:
                    pass
        for t in processing_threads + [bs_send_thread]:
            if t:
                try:
                    t.join(0.1)
                except:
                    pass
        sys.exit(exit_code)


def store_pipeline(stop_event, statistics, parameter_queue,
                   cam_client, pipeline_config, output_stream_port, background_manager, user_scripts_manager=None):

    def no_client_action():
        nonlocal  parameters
        if parameters["no_client_timeout"] > 0:
            _logger.warning("No client connected to the pipeline stream for %d seconds. Closing instance. %s" %
                        (config.MFLOW_NO_CLIENTS_TIMEOUT, log_tag))
            stop_event.set()

    source = None
    sender = None
    log_tag = "store_pipeline"
    exit_code = 0

    parameters = pipeline_config.get_configuration()
    if parameters.get("no_client_timeout") is None:
        parameters["no_client_timeout"] = config.MFLOW_NO_CLIENTS_TIMEOUT
    module = parameters.get("module", None)

    try:
        init_statistics(statistics)

        camera_name = pipeline_config.get_camera_name()
        stream_image_name = camera_name + config.EPICS_PV_SUFFIX_IMAGE
        log_tag = " [" + str(camera_name) + " | " + str(pipeline_config.get_name()) + ":" + str(output_stream_port) + "]"

        camera_stream_address, source_host, source_port, source_mode = resolve_camera_source(cam_client, pipeline_config)
        _logger.warning("Connecting to camera stream address %s. %s" % (camera_stream_address, log_tag))
        source = create_source(camera_stream_address, mode=source_mode)
        source.connect()

        _logger.debug("Opening output stream on port %d. %s", output_stream_port,  log_tag)

        sender = Sender(port=output_stream_port, mode=PUSH,
                        data_header_compression=config.CAMERA_BSREAD_DATA_HEADER_COMPRESSION, block=False)
        sender.open(no_client_action=no_client_action, no_client_timeout=parameters["no_client_timeout"]
                    if parameters["no_client_timeout"] > 0 else sys.maxsize)
        init_sender(sender, parameters)

        # TODO: Register proper channels.
        # Indicate that the startup was successful.
        stop_event.clear()

        _logger.debug("Transceiver started. %s" % log_tag)
        counter = 1

        while not stop_event.is_set():
            try:
                data = source.receive()
                set_statistics(statistics, sender, data.statistics.total_bytes_received if data else statistics.total_bytes, 1 if data else 0)

                if module:
                    if counter < module:
                        counter = counter + 1
                        continue
                    else:
                        counter = 1

                # In case of receiving error or timeout, the returned data is None.
                if data is None:
                    continue

                forward_data = {stream_image_name: data.data.data["image"].value}

                pulse_id = data.data.pulse_id
                timestamp = (data.data.global_timestamp, data.data.global_timestamp_offset)

                send(sender, forward_data, timestamp, pulse_id, parameters, statistics)

            except:
                _logger.exception("Could not process message. %s" % log_tag)
                stop_event.set()

        _logger.info("Stopping transceiver. %s" % log_tag)

    except:
        _logger.exception("Exception while trying to start the receive and process thread. %s" % log_tag)
        exit_code = 1
        raise

    finally:
        if source:
            source.disconnect()

        if sender:
            try:
                sender.close()
            except:
                pass
        sys.exit(exit_code)


def stream_pipeline(stop_event, statistics, parameter_queue,
                   cam_client, pipeline_config, output_stream_port, background_manager, user_scripts_manager=None):

    source = None
    sender = None
    log_tag = "stream_pipeline"
    exit_code = 0

    parameters = get_pipeline_parameters(pipeline_config)

    if pipeline_config.get_input_stream():
        bsread_address = pipeline_config.get_input_stream()
        bsread_mode = pipeline_config.get_input_mode()
        bsread_channels = None
    else:
        bsread_address = parameters.get("bsread_address")
        bsread_mode = parameters.get("bsread_mode")
        bsread_channels = parameters.get("bsread_channels")


    dispatcher_url, dispatcher_verify_request, dispatcher_disable_compression = get_dispatcher_parameters(parameters)

    try:

        init_statistics(statistics)
        log_tag = " ["  + str(pipeline_config.get_name()) + ":" + str(output_stream_port) + "]"

        _logger.debug("Connecting to stream %s. %s" % (str(bsread_address), str(bsread_channels)))

        if bsread_address:
            bsread_host, bsread_port = get_host_port_from_stream_address(bsread_address)
            bsread_mode = SUB if bsread_mode == "SUB" else PULL
        else:
            bsread_host, bsread_port =None, 9999
            bsread_mode = PULL if bsread_mode == "PULL" else SUB

        if bsread_channels is not None:
            if type(bsread_channels) != list:
                bsread_channels = json.loads(bsread_channels)
            if len(bsread_channels)==0:
                bsread_channels = None
        sender = create_sender(parameters, output_stream_port, stop_event, log_tag)

        # Indicate that the startup was successful.
        stop_event.clear()

        _logger.debug("Transceiver started. %s" % log_tag)

        with bssource(host=bsread_host,
                      port=bsread_port,
                      mode=bsread_mode,
                      channels=bsread_channels,
                      receive_timeout=config.PIPELINE_RECEIVE_TIMEOUT,
                      dispatcher_url = dispatcher_url,
                      dispatcher_verify_request = dispatcher_verify_request,
                      dispatcher_disable_compression = dispatcher_disable_compression) as stream:
            while not stop_event.is_set():
                try:
                    while not parameter_queue.empty():
                        new_parameters = parameter_queue.get()
                        pipeline_config.set_configuration(new_parameters)
                        parameters = get_pipeline_parameters(pipeline_config)

                    data = stream.receive()
                    set_statistics(statistics, sender,data.statistics.total_bytes_received if data else statistics.total_bytes, 1 if data else 0)
                    if not data or stop_event.is_set():
                        continue

                    stream_data = OrderedDict()
                    pulse_id = data.data.pulse_id
                    timestamp = (data.data.global_timestamp, data.data.global_timestamp_offset)
                    try:
                        for key, value in data.data.data.items():
                            stream_data[key] = value.value
                        function = get_function(parameters, user_scripts_manager, log_tag)
                        if function is None:
                            continue
                        stream_data = function(stream_data, pulse_id, timestamp, parameters)
                    except Exception as e:
                        _logger.error("Error processing bs buffer: " + str(e) + ". %s" % log_tag)
                        continue
                    send(sender, stream_data, timestamp, pulse_id, parameters, statistics)
                except Exception as e:
                    _logger.exception("Could not process message: " + str(e) + ". %s" % log_tag)
                    stop_event.set()

        _logger.info("Stopping transceiver. %s" % log_tag)

    except:
        _logger.exception("Exception while trying to start the receive and process thread. %s" % log_tag)
        exit_code = 1
        raise

    finally:
        if sender:
            try:
                sender.close()
            except:
                pass
        sys.exit(exit_code)



def custom_pipeline(stop_event, statistics, parameter_queue,
                   cam_client, pipeline_config, output_stream_port, background_manager, user_scripts_manager=None):
    sender = None
    log_tag = "custom_pipeline"
    exit_code = 0

    parameters = get_pipeline_parameters(pipeline_config)
    try:
        init_statistics(statistics)
        log_tag = " ["  + str(pipeline_config.get_name()) + ":" + str(output_stream_port) + "]"
        sender = create_sender(parameters, output_stream_port, stop_event, log_tag)

        function = get_function(parameters, user_scripts_manager, log_tag)
        if function is None:
            raise Exception ("Invalid function")
        max_frame_rate = parameters.get("max_frame_rate")
        sample_interval = (1.0 / max_frame_rate) if max_frame_rate else None

        _logger.debug("Transceiver started. %s" % log_tag)
        # Indicate that the startup was successful.
        stop_event.clear()
        init=True

        while not stop_event.is_set():
            try:
                if sample_interval:
                    start = time.time()
                while not parameter_queue.empty():
                    new_parameters = parameter_queue.get()
                    pipeline_config.set_configuration(new_parameters)
                    parameters = get_pipeline_parameters(pipeline_config)

                stream_data, timestamp, pulse_id, data_size = function(parameters, init)
                init = False
                set_statistics(statistics, sender,statistics.total_bytes + data_size, 1 if stream_data else 0)

                if not stream_data or stop_event.is_set():
                    continue

                send(sender, stream_data, timestamp, pulse_id, parameters, statistics)
                if sample_interval:
                    sleep = sample_interval - (time.time()-start)
                    if (sleep>0):
                        time.sleep(sleep)

            except Exception as e:
                _logger.exception("Could not process message: " + str(e) + ". %s" % log_tag)
                if parameters.get("abort_on_error", config.ABORT_ON_ERROR):
                    stop_event.set()

        _logger.info("Stopping transceiver. %s" % log_tag)

    except:
        _logger.exception("Exception while trying to start the receive and process thread. %s" % log_tag)
        exit_code = 1
        raise

    finally:
        if sender:
            try:
                sender.close()
            except:
                pass
        sys.exit(exit_code)


pipeline_name_to_pipeline_function_mapping = {
    config.PIPELINE_TYPE_PROCESSING: processing_pipeline,
    config.PIPELINE_TYPE_STORE: store_pipeline,
    config.PIPELINE_TYPE_STREAM: stream_pipeline,
    config.PIPELINE_TYPE_CUSTOM: custom_pipeline
}


def get_pipeline_function(pipeline_type_name):
    if pipeline_type_name not in pipeline_name_to_pipeline_function_mapping:
        raise ValueError("pipeline_type '%s' not present in mapping. Available: %s." %
                         (pipeline_type_name, list(pipeline_name_to_pipeline_function_mapping.keys())))

    return pipeline_name_to_pipeline_function_mapping[pipeline_type_name]
