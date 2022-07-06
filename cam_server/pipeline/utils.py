from logging import getLogger
from importlib import import_module
from imp import load_source
import time
import sys
import json
import os
from collections import deque
import threading
from threading import Thread, RLock
import multiprocessing
import numpy
from bsread import source as bssource

from bsread import Source, PUB, SUB, PUSH, PULL, DEFAULT_DISPATCHER_URL
from bsread.sender import Sender
from cam_server import config
from cam_server.pipeline.data_processing.processor import process_image as default_image_process_function
from cam_server.utils import get_host_port_from_stream_address, on_message_sent, get_statistics, update_statistics, init_statistics, MaxLenDict, get_clients
from cam_server.writer import WriterSender, UNDEFINED_NUMBER_OF_RECORDS
from cam_server.ipc import IpcSource

_logger = getLogger(__name__)
_parameters = {}
_parameter_queue = None
_user_scripts_manager = None
_parameters_post_proc = None
_pipeline_config = None

sender = None
source = None
camera_host = None
camera_port = None
cam_client = None
functions = {}
log_tag = ""
number_processing_threads = 0
thread_buffers = []
multiprocessed = False
received_pids = None
tx_lock = None
processing_threads = []
message_buffer, message_buffer_send_thread  = None, None
debug = False
current_pid, former_pid = None, None
camera_timeout = None
stream_timeout = None
pause = False
pid_range = None
downsampling = None
downsampling_counter= None
function = None
message_buffer_size = None
thread_exit_code=0
last_rcvd_timestamp = time.time()


class ProcessingCompleted(Exception):
     pass


class SourceTimeout(Exception):
    pass

def set_log_tag(tag):
    global log_tag
    log_tag = tag

def get_log_tag(tag):
    return log_tag

def init_sender(sender, pipeline_parameters):
    sender.record_count = 0
    sender.enforce_pid = pipeline_parameters.get("enforce_pid")
    sender.last_pid=-1
    sender.data_format = None
    create_header = pipeline_parameters.get("create_header")
    if create_header in (True,"always"):
        sender.create_header = True
    elif create_header in (False, "once"):
        sender.create_header = False
    else:
        sender.create_header = None
    sender.records=pipeline_parameters.get("records")

def create_sender(output_stream_port, stop_event):
    global sender
    sender = None
    pars = get_parameters()
    def no_client_action():
        global sender
        nonlocal pars
        if pars["no_client_timeout"] > 0:
            _logger.warning("No client connected to the pipeline stream for %d seconds. Closing instance. %s",
                            pars["no_client_timeout"], log_tag)
            stop_event.set()
            if sender:
                if pars["mode"] == "PUSH" and pars["block"]:
                    _logger.warning("Killing the process: cannot stop gracefully if sender is blocking")
                    os._exit(0)

    if pars["mode"] == "FILE":
        file_name = pars["file"]
        records = pars.get("records")
        sender = WriterSender(output_file=file_name,
                              number_of_records=records if records else UNDEFINED_NUMBER_OF_RECORDS,
                              layout=pars["layout"],
                              save_local_timestamps=pars["localtime"],
                              change=pars["change"],
                              attributes={})
    else:
        sender = Sender(port=output_stream_port,
                        mode=PUSH if (pars["mode"] == "PUSH") else PUB,
                        queue_size=pars["queue_size"],
                        block=pars["block"],
                        data_header_compression=config.CAMERA_BSREAD_DATA_HEADER_COMPRESSION)
    sender.open(no_client_action=no_client_action, no_client_timeout=pars["no_client_timeout"]
                if pars["no_client_timeout"] > 0 else sys.maxsize)
    init_sender(sender, pars)
    return sender

def get_sender():
    global sender
    return sender

def check_records(sender):
    records =get_parameters().get("records")
    if records:
        sender.record_count = sender.record_count + 1
        if sender.record_count >= records:
            raise ProcessingCompleted("Reached number of records: " + str(records))

def send(sender, data, timestamp, pulse_id):
    if sender is None:
        sender = get_sender()
    try:
        if sender.enforce_pid:
            if pulse_id <= sender.last_pid:
                _logger.warning("Sending invalid PID: %d - last: %d" ". %s" % (pulse_id, sender.last_pid, log_tag))
                return
            sender.last_pid = pulse_id
        if sender.create_header == True:
            check_header = True
        elif sender.create_header == False:
            check_header = (sender.data_format is None)
            sender.data_format = True
        else:
            try:
                data_format = {k: ((v.shape, v.dtype) if isinstance(v, numpy.ndarray) else
                                (len(v) if isinstance(v, list) else type(v))) for k, v in data.items()}
                check_header = data_format != sender.data_format
            except Exception as ex:
                _logger.warning("Exception checking header change: " + str(ex) + ". %s" % log_tag)
                sender.data_format = None
                check_header = True
            if check_header:
                sender.data_format = data_format
        sender.send(data=data, timestamp=timestamp, pulse_id=pulse_id, check_data=check_header)
        on_message_sent()
        if sender.records:
            check_records(sender)
    except Exception as ex:
        _logger.exception("Exception in the sender: " + str(ex) + ". %s" % log_tag)
        raise


def get_parameters():
    return _parameters

def init_pipeline_parameters(pipeline_config, parameter_queue =None, user_scripts_manager=None, post_processsing_function=None):
    global _parameters, _parameter_queue, _user_scripts_manager, _parameters_post_proc, _pipeline_config
    global pause, pid_range, downsampling, downsampling_counter, function, debug, camera_timeout, stream_timeout

    parameters = pipeline_config.get_configuration()
    if parameters.get("no_client_timeout") is None:
        parameters["no_client_timeout"] = config.MFLOW_NO_CLIENTS_TIMEOUT

    if parameters.get("queue_size") is None:
        parameters["queue_size"] = config.PIPELINE_DEFAULT_QUEUE_SIZE

    if parameters.get("mode") is None:
        if pipeline_config.get_pipeline_type() ==config.PIPELINE_TYPE_FANOUT:
            parameters["mode"] = "PUSH"
        else:
            parameters["mode"] = config.PIPELINE_DEFAULT_MODE

    if parameters.get("block") is None:
        parameters["block"] = config.PIPELINE_DEFAULT_BLOCK

    if parameters.get("pid_range"):
        try:
            parameters["pid_range"] = int(parameters.get("pid_range")[0]), int(parameters.get("pid_range")[1])
        except:
            parameters["pid_range"] = None

    debug = parameters.get("debug", False)
    camera_timeout = parameters.get("camera_timeout", 10.0)
    pause = parameters.get("pause", False)
    pid_range = parameters.get("pid_range", None)
    downsampling = parameters.get("downsampling", None)
    downsampling_counter = sys.maxsize  # The first is always sent
    stream_timeout = parameters.get("stream_timeout", 10.0)
    function = get_function(parameters, user_scripts_manager)

    _parameter_queue = parameter_queue
    _parameters = parameters
    #_parameters.clear()
    #_parameters.update(parameters)
    _user_scripts_manager=user_scripts_manager
    _parameters_post_proc = post_processsing_function
    _pipeline_config = pipeline_config
    return parameters


def check_parameters_changes():
    global _parameters, _parameter_queue, _user_scripts_manager, _parameters_post_proc, _pipeline_config
    changed = False
    while not _parameter_queue.empty():
        new_parameters = _parameter_queue.get()
        _pipeline_config.set_configuration(new_parameters)
        changed = True
    if changed:
        init_pipeline_parameters(_pipeline_config, _parameter_queue, _user_scripts_manager, _parameters_post_proc)
        if _parameters_post_proc:
            return _parameters_post_proc()
        return get_parameters()
    return None

def abort_on_error():
    pars = get_parameters()
    return pars.get("abort_on_error", config.ABORT_ON_ERROR)

def abort_on_timeout():
    pars = get_parameters()
    return pars.get("abort_on_timeout", config.ABORT_ON_TIMEOUT)

def timeout_count():
    pars = get_parameters()
    return pars.get("timeout_count", config.TIMEOUT_COUNT)

def assert_function_defined():
    global function
    if function is None:
        raise Exception("Undefined processing function")


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


def get_function(pipeline_parameters, user_scripts_manager):
    if pipeline_parameters is None:
        pipeline_parameters = get_parameters()
    name = pipeline_parameters.get("function")
    if not name:
        if pipeline_parameters.get("pipeline_type") == config.PIPELINE_TYPE_PROCESSING:
            return default_image_process_function
        return None
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
        #import traceback
        #traceback.print_exc()
        _logger.exception("Could not import function: %s. %s" % (str(name), log_tag))
        return None

def create_source(camera_stream_address, receive_timeout=config.PIPELINE_RECEIVE_TIMEOUT, mode=SUB):
    source_host, source_port = get_host_port_from_stream_address(camera_stream_address)
    if camera_stream_address.startswith("ipc"):
        return IpcSource(address=camera_stream_address,receive_timeout=receive_timeout, mode=mode)
    else:
        return Source(host=source_host, port=source_port, receive_timeout=receive_timeout, mode=mode)


def resolve_camera_source(cam_client, pars=None):
    if pars is None:
        pars = get_parameters()
    source_mode = SUB
    input_stream = pars.get("input_stream")
    if input_stream:
        camera_stream_address = input_stream
        source_mode = PULL if pars.get("input_mode", "SUB") == "PULL" else SUB
    else:
        camera_stream_address = cam_client.get_instance_stream(pars.get("camera_name"))
    source_host, source_port = get_host_port_from_stream_address(camera_stream_address)
    return camera_stream_address, source_host, source_port, source_mode

def is_camera_source():
    return cam_client is not None

def connect_to_camera(_cam_client):
    global source, camera_host, camera_port, current_pid, former_pid, cam_client
    cam_client=_cam_client
    current_pid, former_pid = None, None
    camera_stream_address, source_host, source_port, source_mode = resolve_camera_source(cam_client)
    _logger.warning("Connecting to camera stream address %s. %s" % (camera_stream_address, log_tag))
    if (source is None) or (source_host != camera_host) or (source_port != camera_port):
        if source:
            try:
                source.disconnect()
                source = None
            except:
                pass
        source = create_source(camera_stream_address, mode=source_mode)
        source.connect()
        camera_host, camera_port = source_host, source_port
    return source

def has_stream():
    pars = get_parameters()
    return pars.get("input_stream") or pars.get("bsread_address") or (pars.get("bsread_channels") is not None)


def connect_to_stream():
    global source
    pars = get_parameters()

    input_stream = pars.get("input_stream")
    if input_stream:
        bsread_address = input_stream
        bsread_channels = None
    else:
        bsread_address = pars.get("bsread_address")
        bsread_channels = pars.get("bsread_channels")
    bsread_mode = pars.get("input_mode", "SUB")

    _logger.debug("Connecting to stream %s. %s" % (str(bsread_address), str(bsread_channels)))

    receive_timeout = int(pars.get("receive_timeout", config.PIPELINE_RECEIVE_TIMEOUT))

    dispatcher_url, dispatcher_verify_request, dispatcher_disable_compression = get_dispatcher_parameters(pars)

    if bsread_address:
        bsread_host, bsread_port = get_host_port_from_stream_address(bsread_address)
        bsread_mode = SUB if bsread_mode == "SUB" else PULL
    else:
        bsread_host, bsread_port = None, 9999
        bsread_mode = PULL if bsread_mode == "PULL" else SUB

    if bsread_channels is not None:
        if type(bsread_channels) != list:
            bsread_channels = json.loads(bsread_channels)
        if len(bsread_channels) == 0:
            bsread_channels = None

    ret = bssource(  host=bsread_host,
                      port=bsread_port,
                      mode=bsread_mode,
                      channels=bsread_channels,
                      receive_timeout=receive_timeout,
                      dispatcher_url = dispatcher_url,
                      dispatcher_verify_request = dispatcher_verify_request,
                      dispatcher_disable_compression = dispatcher_disable_compression)
    if not is_camera_source():
        source = ret.source
    return ret

def connect_to_source(cam_client):
    global _pipeline_config
    camera_name = _pipeline_config.get_camera_name()
    if camera_name:
        source = connect_to_camera(cam_client)
    else:
        source = connect_to_stream()
        source.source.connect()
    return source



def send_data(processed_data, global_timestamp, pulse_id, message_buffer = None):
    if processed_data is not None:
        pars = get_parameters()
        # Requesting subset of the data
        include = pars.get("include")
        if include:
            aux = {}
            for key in include:
                aux[key] = processed_data.get(key)
            processed_data = aux
        exclude = pars.get("exclude")
        if exclude:
            for field in exclude:
                processed_data.pop(field, None)

        if message_buffer:
            message_buffer.append((processed_data, global_timestamp, pulse_id))
        else:
            send(sender, processed_data, global_timestamp, pulse_id)
        #_logger.debug("Sent PID %d" % (pulse_id,))


def receive_stream(camera=False):
    global last_rcvd_timestamp, pause, pid_range, downsampling, downsampling_counter, camera_timeout, stream_timeout
    pulse_id = global_timestamp = data = None
    rx = source.receive()

    if rx:
        msg=rx.data
        pulse_id = msg.pulse_id
        global_timestamp = (msg.global_timestamp, msg.global_timestamp_offset)
        data = msg.data

        frame_shape = None
        if camera:
            image = data["image"].value
            if image is not None:
                frame_shape = str(image.shape[1]) + "x" + str(image.shape[0]) + "x" + str(image.itemsize)
        last_rcvd_timestamp = time.time()
        if debug:
            global former_pid, current_pid
            if (former_pid is not None) and (current_pid is not None):
                modulo = current_pid - former_pid
                expected = current_pid + modulo
                if pulse_id != expected:
                    lost = int((pulse_id - expected)/modulo)
                    if lost > 0:
                        _logger.warning("Unexpected PID: " + str(pulse_id) + " -  last: " + str(former_pid) + ", cur:" + str(current_pid)+ ", lost:" + str(lost))
                    else:
                        _logger.debug("Newer PID: " + str(pulse_id) + " -  last: " + str(former_pid) + ", cur:" + str(current_pid))
                    current_pid, former_pid = None, None
            former_pid = current_pid
            current_pid = pulse_id
        update_statistics(sender, rx.statistics.total_bytes_received, 1, frame_shape)

        if pause:
            return pulse_id, global_timestamp, None

        if pid_range:
            if (pid_range[0] <= 0) or (pulse_id < pid_range[0]):
                return pulse_id, global_timestamp, None
            elif (pid_range[1] > 0) and (pulse_id > pid_range[1]):
                _logger.warning("Reached end of pid range: stopping pipeline")
                raise ProcessingCompleted("End of pid range")

        # Check downsampling parameter
        if downsampling:
            downsampling_counter += 1
            if downsampling_counter > downsampling:
                downsampling_counter = 0
            else:
                return pulse_id, global_timestamp, None
    else:
        update_statistics(sender, 0, 0, None)
        if abort_on_timeout():
            if (stream_timeout > 0) and (time.time() - last_rcvd_timestamp) > stream_timeout:
                 raise SourceTimeout("Stream Timeout")
        if camera:
            if camera_timeout:
                if (camera_timeout > 0) and (time.time() - last_rcvd_timestamp) > camera_timeout:
                    _logger.warning("Camera timeout. %s" % log_tag)
                    last_rcvd_timestamp = time.time()
                    # Try reconnecting to the camera. If fails raise exception and stops pipeline.
                    connect_to_camera(cam_client)


    return pulse_id, global_timestamp, data

def get_next_thread_index():
    global number_processing_threads, processing_thread_index
    index = processing_thread_index
    processing_thread_index = processing_thread_index + 1
    if processing_thread_index >= number_processing_threads:
        processing_thread_index = 0
    return index


def process_data(processing_function, pulse_id, global_timestamp, *args):
    global number_processing_threads, received_pids, tx_lock, thread_buffers, message_buffer, message_buffer_size, load_balancing

    if (not message_buffer_size) and (number_processing_threads > 0):
        if multiprocessed:
            index = get_next_thread_index()
            thread_buffer = thread_buffers[index]
            received_pids.put(pulse_id)
            thread_buffer.put((pulse_id, global_timestamp, message_buffer, get_parameters(), *args))
        else:
            lost_pid = None

            with tx_lock:
                load = [len(thread_buffer) for thread_buffer in thread_buffers]
                # Gets the thread with lower load.
                # If load ist he same, then gets next in order, in order to try to make homogenous lost of pids in case of thread buffer full.
                min_load, max_load = min(load), max(load)
                if min_load == max_load:
                    index = get_next_thread_index()
                else:
                    index = load.index(min_load)

                thread_buffer = thread_buffers[index]
                if len(thread_buffer) >= thread_buffer.maxlen:
                        lost = thread_buffer.popleft()
                        lost_pid = lost[0]
                        try:
                            received_pids.remove(lost_pid)
                        except:
                            pass
                thread_buffer.append((pulse_id, global_timestamp, message_buffer, *args))
                received_pids.append(pulse_id)
            if lost_pid is not None:
                if debug:
                    _logger.error("Thread %d buffer full: lost PID %d " % (index, lost_pid))

        return
    processed_data = processing_function(pulse_id, global_timestamp, function, *args)
    if processed_data is not None:
        send_data(processed_data, global_timestamp, pulse_id, message_buffer)


def setup_sender(output_port, stop_event, pipeline_processing_function=None, user_scripts_manager=None):
    global number_processing_threads, multiprocessed, processing_thread_index, received_pids, processing_threads, message_buffer_send_thread, tx_lock, thread_buffers, message_buffer_size
    pars = get_parameters()
    number_processing_threads = pars.get("processing_threads", 0)
    thread_buffers = None if number_processing_threads == 0 else []
    multiprocessed = pars.get("multiprocessing", False)

    message_buffer_size = pars.get("buffer_size")
    if message_buffer_size:
        message_buffer = deque(maxlen=message_buffer_size)
        message_buffer_send_thread = Thread(target=message_buffer_send_task, args=(message_buffer, output_port, stop_event))
        message_buffer_send_thread.start()
    else:
        if number_processing_threads > 0:
            processing_thread_index = 0
            thread_buffer_size = pars.get("thread_buffer_size", 20)
            send_buffer_size = pars.get("send_buffer_size", 40)
            if multiprocessed:
                tx_lock = multiprocessing.Lock()

                if pars.get("processing_manager", False):
                    received_pids = multiprocessing.Manager().Queue()
                    tx_queue = multiprocessing.Manager().Queue(send_buffer_size)
                else:
                    received_pids = multiprocessing.Queue()
                    tx_queue = multiprocessing.Queue(send_buffer_size)
                spawn_send_thread = pars.get("spawn_send_thread", True)
                message_buffer_send_thread = multiprocessing.Process(target=process_send_task, args=(output_port, tx_queue, received_pids, spawn_send_thread, stop_event, pars, log_tag))
                message_buffer_send_thread.start()
                for i in range(number_processing_threads):
                    thread_buffer = multiprocessing.Queue(thread_buffer_size)
                    thread_buffers.append(thread_buffer)
                    processing_thread = multiprocessing.Process(target=process_task, args=( #TODO change config in processes
                    pipeline_processing_function, thread_buffer, tx_queue, stop_event, i, user_scripts_manager, pars, log_tag))
                    processing_threads.append(processing_thread)
                    processing_thread.start()
            else:
                tx_lock = RLock()
                received_pids = deque()
                tx_buffer = MaxLenDict(maxlen=(send_buffer_size))
                message_buffer_send_thread = Thread(target=thread_send_task,args=(output_port, tx_buffer, tx_lock, received_pids, stop_event))
                message_buffer_send_thread.start()
                for i in range(number_processing_threads):
                    thread_buffer = deque(maxlen=thread_buffer_size)
                    thread_buffers.append(thread_buffer)
                    processing_thread = Thread(target=thread_task, args=(
                    pipeline_processing_function, thread_buffer, tx_buffer, tx_lock, received_pids, stop_event, i))
                    processing_threads.append(processing_thread)
                    processing_thread.start()
        else:
            create_sender(output_port, stop_event)


#Message buffering
def message_buffer_send_task(message_buffer, output_port, stop_event):
    pars = get_parameters()
    _logger.info("Start message buffer send thread")
    create_sender(output_port, stop_event)
    try:
        while not stop_event.is_set():
            if len(message_buffer) == 0:
                time.sleep(0.001)
            else:
                (processed_data, timestamp, pulse_id) = message_buffer.popleft()
                send(sender, processed_data, timestamp, pulse_id)

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


#Multi-threading
def thread_task(process_function, thread_buffer, tx_buffer, tx_lock, received_pids, stop_event, index):
    global thread_exit_code, debug
    _logger.info("Start processing thread %d: %d" % (index,threading.get_ident()))
    try:
        while not stop_event.is_set():
            with tx_lock:
                try:
                    (pulse_id, global_timestamp, message_buffer, *args) = msg = thread_buffer.popleft()
                    if pulse_id not in received_pids:
                        msg = None
                except IndexError:
                    msg = None
            if msg is None:
                time.sleep(0.001)
                continue
            processed_data = process_function(pulse_id, global_timestamp, function, *args)
            if processed_data is None:
                if debug:
                    _logger.info ("Error processing PID %d at thread %d" % (pulse_id, index))
                try:
                    with tx_lock:
                        received_pids.remove(pulse_id)
                except:
                    _logger.warning("Error removing PID %d at thread %d" % (pulse_id, index))
            else:
                lost_pid = None
                with tx_lock:
                    if pulse_id in received_pids:
                        if len(tx_buffer) >= tx_buffer.maxlen:
                            lost_pid = min(tx_buffer.keys())
                            del tx_buffer[lost_pid]
                            try:
                                received_pids.remove(lost_pid)
                            except:
                                pass
                        tx_buffer[pulse_id] = (processed_data, global_timestamp, pulse_id, message_buffer)
                if lost_pid:
                    if debug:
                        _logger.info("Send buffer full - removing oldest PID: %d at thread %d" % (pulse_id, index))

    except Exception as e:
        thread_exit_code = 2
        _logger.error("Error on processing thread %d: %s" % (index, str(e)))
    finally:
        stop_event.set()
        _logger.info("Exit processing thread %d" % index)


def thread_send_task(output_port, tx_buffer, tx_lock, received_pids, stop_event):
    _logger.info("Start threaded processing send thread")
    sender = create_sender(output_port, stop_event)
    pid = None
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
                        tx = tx_buffer.pop(pid)  # tx=(processed_data, global_timestamp, pulse_id, message_buffer)
                    else:
                        if size >= tx_buffer.maxlen:
                            pid = received_pids.popleft()
                            popped = True
            if tx is not None:
                send_data(*tx)
            else:
                if popped:
                    if debug:
                        _logger.error("Removed timed-out processing -  Pulse ID: " + str(pid))
                time.sleep(0.01)
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


# Multi-processing
def process_task(process_function, thread_buffer, tx_queue, stop_event, index, user_scripts_manager, pipeline_config, ltag):
    global log_tag
    global _parameters
    log_tag = ltag
    _logger.info("Start process %d: %d" % (index,os.getpid()) )

    try:
        while not stop_event.is_set():
            try:
                (pulse_id, global_timestamp, message_buffer, pars, *args) = thread_buffer.get(False)
                _parameters = pars
            except Exception as e:
                time.sleep(0.001)
                continue
            #check_parameters_changes()
            function = get_function(pars, user_scripts_manager)
            processed_data = process_function(pulse_id, global_timestamp, function, *args)
            try:
                tx_queue.put((processed_data, global_timestamp, pulse_id, message_buffer), False)
            except Exception as e:
                _logger.error("Error adding to tx buffer %d: %s" % (index, str(e)))
            #_logger.info("Processed message %d in process %d" % (pulse_id, index))


    except Exception as e:
        _logger.error("Error on process %d: %s" % (index, str(e)))
        #import traceback
        #traceback.print_exc()
    finally:
        stop_event.set()
        _logger.info("Exit process  %d" % index)
        sys.exit(0)


def process_send_task(output_port, tx_queue, received_pids_queue, spawn_send_thread, stop_event, pars, ltag):
    global log_tag
    log_tag = ltag

    sender = None
    _parameters = pars
    _logger.info("Start send process")
    received_pids = deque()
    tx_buffer = MaxLenDict(maxlen=(tx_queue._maxsize))

    if spawn_send_thread:
        tx_buffer_lock = RLock()
        send_thread = Thread(target=thread_send_task,
                                        args=(output_port, tx_buffer, tx_buffer_lock, received_pids, stop_event))
        send_thread.start()
    else:
        sender = create_sender(output_port, stop_event)

    last_msg_timestamp = time.time()
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
                                pid = received_pids.popleft()
                                if get_parameters().get("debug"):
                                    _logger.error("Timeout processing PID " + str(pid))
                        if tx is not None:
                            send_data(processed_data, global_timestamp, pulse_id, message_buffer)
                        else:
                            break
                # When multiprocessed cannot access sender object from main process
                if ((time.time() - last_msg_timestamp) > 1.0):
                    get_statistics().num_clients = get_clients(get_sender())
                    last_msg_timestamp = time.time()

    except Exception as e:
        _logger.error("Error send process" + str(e))
    finally:
        stop_event.set()
        if sender:
            try:
                sender.close()
            except:
                pass
        _logger.info("Exit send process")
        sys.exit(0)


def cleanup():
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
    for t in processing_threads:
        if t:
            try:
                t.join(0.1)
            except:
                pass
    if thread_exit_code!=0:
        sys.exit(thread_exit_code)
