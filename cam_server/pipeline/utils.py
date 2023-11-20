import json
import logging
import multiprocessing
import os
import sys
import signal
import threading
import time
from collections import deque
from imp import load_source
from importlib import import_module
from threading import Thread, RLock

import numpy
from bsread import Source, PUB, SUB, PUSH, PULL, DEFAULT_DISPATCHER_URL
from bsread import source as bssource
from bsread.sender import Sender, BIND, CONNECT

from cam_server import config, merger
from cam_server.ipc import IpcSource
from cam_server.loader import load_module
from cam_server.otel import otel_get_tracer, otel_get_meter, otel_setup_logs
from cam_server.pipeline.data_processing.processor import process_image as default_image_process_function
from cam_server.utils import on_message_sent, get_statistics, update_statistics, MaxLenDict, get_clients, setup_instance_logs, timestamp_as_float, set_log_suffix, get_log_suffix
from cam_server.writer import WriterSender, UNDEFINED_NUMBER_OF_RECORDS
from cam_server_client.utils import get_host_port_from_stream_address

_logger = logging.getLogger(__name__)

if config.TELEMETRY_ENABLED:
    from opentelemetry.sdk.trace import Status, StatusCode
    otel_setup_logs()
    tracer = otel_get_tracer()
    meter = otel_get_meter()

    run_counter = meter.create_counter("run_counter", description="Number of runs of the pipeline by status")
    run_timespan = meter.create_histogram(name="run_timespan", unit="seconds", description="Pipeline processing time")

_parameters = {}
_parameter_queue = None
_logs_queue = None
_user_scripts_manager = None
_parameters_post_proc = None
_pipeline_config = None
_message_buffer = None

sender = None
source = None
pid_buffer = None
pid_buffer_size = None
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
stream_failed= False
pause = False
pid_range = None
downsampling = None
downsampling_counter= None
function = None
message_buffer_size = None
thread_exit_code=0
thread_exit_code=0
max_frame_rate=None
last_sent_timestamp=0
camera_name = None
output_stream_port = 0
pipeline_name = None


last_rcvd_timestamp = time.time()


class ProcessingCompleted(Exception):
     pass


class SourceTimeout(Exception):
    pass

def set_log_tag(tag):
    global log_tag
    log_tag = tag

def get_log_tag():
    return log_tag

def init_sender(sender, pipeline_parameters):
    sender.record_count = 0
    sender.enforce_pid = pipeline_parameters.get("enforce_pid")
    sender.enforce_timestamp = pipeline_parameters.get("enforce_timestamp")
    sender.check_timestamp = pipeline_parameters.get("check_timestamp")
    sender.last_pid = -1
    sender.last_timestamp_float = -1
    sender.last_timestamp = None
    sender.data_format = None
    sender.header_changes = 0
    create_header = pipeline_parameters.get("create_header")
    if create_header in (True,"always"):
        sender.create_header = True
    elif create_header in (False, "once"):
        sender.create_header = False
    else:
        sender.create_header = None
    sender.allow_type_changes = pipeline_parameters.get("allow_type_changes", True)
    sender.allow_shape_changes = pipeline_parameters.get("allow_shape_changes", True)
    sender.records = pipeline_parameters.get("records")

def create_sender(output_stream_port, stop_event):
    global sender, pid_buffer, pid_buffer_size
    sender = None
    pars = get_parameters()
    def no_client_action():
        global sender
        nonlocal pars
        if pars["no_client_timeout"] > 0:
            _logger.warning("No client connected to the pipeline stream for %d seconds. Closing instance.%s" %(pars["no_client_timeout"], get_log_tag()))
            stop_event.set()
            if sender:
                if pars["mode"] == "PUSH" and pars["block"]:
                    _logger.warning("Killing the process: cannot stop gracefully if sender is blocking.%s" % get_log_tag())
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
        output_stream = pars.get("output_stream", None)
        address = "tcp://*"
        connect_type = BIND
        if output_stream:
            connect_type = CONNECT
            address, output_stream_port = get_host_port_from_stream_address(output_stream)
            address = "tcp://" + address

        sender = Sender(port=output_stream_port,
                        address=address,
                        conn_type=connect_type,
                        mode=PUSH if (pars["mode"] == "PUSH") else PUB,
                        queue_size=pars["queue_size"],
                        block=pars["block"],
                        data_header_compression=pars["data_header_compression"],
                        data_compression=pars["data_compression"]
                        )
        if pars["data_compression"]:
            _logger.info("Created sender with data compression: %s.%s" %(str(pars["data_compression"]), get_log_tag()))
        if pars["data_header_compression"]:
            _logger.info("Created sender with header compression: %s.%s" % (str(pars["data_header_compression"]), get_log_tag()))
    sender.open(no_client_action=no_client_action, no_client_timeout=pars["no_client_timeout"]
                if pars["no_client_timeout"] > 0 else sys.maxsize)
    init_sender(sender, pars)
    if pars.get("pid_buffer",0) > 1:
        pid_buffer_size = pars.get("pid_buffer")
        pid_buffer = {}
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
        if sender.check_timestamp:
            if (pulse_id % 1000000) != (timestamp[1] % 1000000):
                _logger.info("Received incompatible Timestamp: %s  PID: %d %s" % (str(timestamp), pulse_id, get_log_tag()))
                return
        if sender.enforce_pid:
            if pulse_id <= sender.last_pid:
                _logger.warning("Sending invalid PID: %d - last: %d %s" % (pulse_id, sender.last_pid, get_log_tag()))
                return
        if sender.enforce_timestamp:
            timestamp_float = timestamp_as_float(timestamp)
            if timestamp_float <= sender.last_timestamp_float:
                _logger.warning("Sending invalid Timestamp: %s %f - last: %s %f PID: %d - last: %d %s" % (str(timestamp), timestamp_float, str(sender.last_timestamp), sender.last_timestamp_float, pulse_id, sender.last_pid, get_log_tag()))
                return
            sender.last_timestamp = timestamp
            sender.last_timestamp_float = timestamp_float
        sender.last_pid = pulse_id
        if sender.create_header == True:
            check_header = True
        elif sender.create_header == False:
            check_header = (sender.data_format is None)
            sender.data_format = True
        else:
            try:
                def get_desc(v):
                    if isinstance(v, list):  # Reason lists
                        v = numpy.array(v)
                    if isinstance(v, numpy.ndarray):
                        return v.shape, v.dtype
                    if isinstance(v, numpy.floating):  # Different scalar float types don't change header
                        return float
                    if isinstance(v, numpy.integer):  # Different scalar int types don't change header
                        return int
                    return type(v)


                check_header = False
                #data_format = {k: get_desc(v) for k, v in data.items()}
                #check_header = data_format != sender.data_format
                #sender.data_format = data_format

                if sender.data_format is None:
                    sender.data_format = {}

                msg_keys_to_remove = []
                for k, v in data.items():
                    cur_fmt = sender.data_format.get(k)
                    if v is None:
                        #Value of channel is None: does not recreate header
                        if not cur_fmt:
                            # Only include channel in message type is known
                            msg_keys_to_remove.append(k)
                    else:
                        #Type of channel changed: recreate header
                        fmt = get_desc(v)
                        if fmt != cur_fmt:
                            if cur_fmt:
                                old_shape = cur_fmt[0] if type(cur_fmt) is tuple else 0
                                old_type = cur_fmt[1] if type(cur_fmt) is tuple else cur_fmt
                                new_shape = fmt[0] if type(fmt) is tuple else 0
                                new_type = fmt[1] if type(fmt) is tuple else fmt
                                if old_type != new_type:
                                    if sender.allow_type_changes:
                                        _logger.debug("Channel %s type change: %s to %s.%s" % (k, str(old_type), str(new_type), get_log_tag()))
                                        check_header = True
                                    else:
                                        _logger.warning("Invalid channel %s type change: %s to %s.%s" % (k, str(old_type), str(new_type),  get_log_tag()))
                                        data[k] = None
                                        fmt = cur_fmt
                                if old_shape != new_shape:
                                    if sender.allow_shape_changes:
                                        _logger.debug("Channel %s shape change: %s to %s.%s" % (k, str(old_shape), str(new_shape),  get_log_tag()))
                                        check_header = True
                                    else:
                                        _logger.warning("Invalid channel %s shape change: %s to %s.%s" % (k, str(old_shape), str(new_shape),  get_log_tag()))
                                        data[k] = None
                                        fmt = cur_fmt
                            else:
                                check_header = True
                        sender.data_format[k] = fmt
                for k in msg_keys_to_remove:
                    try:
                        del data[k]
                    except:
                        pass

                #Channels have been removed from the message: recreate the header.
                sender_keys_to_remove = []
                for k in sender.data_format.keys():
                    if not k in data:
                        check_header = True
                        sender_keys_to_remove.append(k)
                for k in sender_keys_to_remove:
                    del sender.data_format[k]


            except Exception as ex:
                _logger.warning("Exception checking header change: " + str(ex) + ".%s" % get_log_tag())
                sender.data_format = None
                data = None
        if data:
            sender.send(data=data, timestamp=timestamp, pulse_id=pulse_id, check_data=check_header)
            if check_header:
                sender.header_changes=sender.header_changes+1
            on_message_sent()
            if sender.records:
                check_records(sender)
    except Exception as ex:
        _logger.exception("Exception in the sender: " + str(ex) + ".%s" % get_log_tag())
        raise


def get_parameters():
    return _parameters


def init_pipeline_parameters(pipeline_config, parameter_queue =None, logs_queue=None, user_scripts_manager=None, post_processsing_function=None, port=None):
    global _parameters, _parameter_queue, _user_scripts_manager, _parameters_post_proc, _pipeline_config
    global pause, pid_range, downsampling, downsampling_counter, function, debug, camera_timeout, stream_timeout, max_frame_rate, last_sent_timestamp
    global camera_name, output_stream_port, pipeline_name
    camera_name = pipeline_config.get_camera_name()
    pipeline_name = pipeline_config.get_name()
    if port is not None:
        output_stream_port = port

    #if camera_name:
    #    set_log_tag(" [" + str(camera_name) + " | " + str(pipeline_config.get_name()) + ":" + str(output_stream_port) + "]")
    #else:
    #    set_log_tag(" [" + str(pipeline_config.get_name()) + ":" + str(output_stream_port) + "]")
    set_log_suffix(" [name:%s]" % (str(pipeline_config.get_name())))
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
    max_frame_rate = parameters.get("max_frame_rate")
    last_sent_timestamp = 0
    downsampling_counter = sys.maxsize  # The first is always sent

    stream_timeout = parameters.get("stream_timeout", 10.0)
    stream_failed = False

    if parameters.get("data_compression", config.PIPELINE_BSREAD_DATA_COMPRESSION):
        if parameters.get("data_compression")  == True:
            parameters["data_compression"] = "bitshuffle_lz4"
    else:
        parameters["data_compression"] = None

    if parameters.get("data_header_compression", config.PIPELINE_BSREAD_DATA_HEADER_COMPRESSION):
        if parameters.get("data_header_compression")  == True:
            parameters["data_header_compression"] = "bitshuffle_lz4"
    else:
        parameters["data_header_compression"] = None

    if parameter_queue is not None:
        _parameter_queue = parameter_queue
    _parameters = parameters
    #_parameters.clear()
    #_parameters.update(parameters)
    if user_scripts_manager is not None:
        _user_scripts_manager = user_scripts_manager
    if post_processsing_function is not None:
        _parameters_post_proc = post_processsing_function
    _pipeline_config = pipeline_config

    if user_scripts_manager and user_scripts_manager.get_lib_home():
        sys.path.append(os.path.abspath(user_scripts_manager.get_lib_home()))

    function = get_function(parameters, user_scripts_manager)
    global _logs_queue
    if _logs_queue is None:
        _logs_queue = logs_queue
        setup_instance_logs(logs_queue)
    return parameters


def check_parameters_changes():
    global _parameters, _parameter_queue, _logs_queue,_user_scripts_manager, _parameters_post_proc, _pipeline_config
    changed = False
    while not _parameter_queue.empty():
        new_parameters = _parameter_queue.get()
        _pipeline_config.set_configuration(new_parameters)
        _logger.info("Configuration update: %s.%s" % (str(new_parameters), get_log_tag()))
        changed = True
    if changed:
        init_pipeline_parameters(_pipeline_config, _parameter_queue, _logs_queue, _user_scripts_manager, _parameters_post_proc)
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
                _logger.info("Importing function: %s.%s" % (name, get_log_tag()))
            else:
                _logger.info("Reloading function: %s.%s" % (name, get_log_tag()))
            module_name = name # 'mod'
            if '/' in name:
                mod = load_source(module_name, name)
            else:
                try:
                    if user_scripts_manager and user_scripts_manager.exists(name):
                        mod = load_source(module_name, user_scripts_manager.get_path(name))
                    else:
                        mod = import_module("cam_server.pipeline.data_processing." + str(name))
                except:
                    #try loading C extension
                    mod = load_module(name, user_scripts_manager.get_home())
            try:
                functions[name] = f = mod.process_image
            except:
                functions[name] = f = mod.process
            pipeline_parameters["reload"] = False
        return f
    except Exception as e:
            #import traceback
            #traceback.print_exc()
            _logger.exception("Could not import function %s: %s.%s" % (str(name), str(e), get_log_tag()))
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
    _logger.warning("Connecting to camera stream address %s.%s" % (camera_stream_address, get_log_tag()))
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
    if pars.get("bsread_address") or (pars.get("bsread_channels") is not None):
        return True
    return False


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

    default_input_mode = "SUB"
    conn_type = CONNECT
    if pars.get("pipeline_type","") in ["fanin",]:
        conn_type = BIND
        default_input_mode = "PULL"

    bsread_mode = pars.get("input_mode", default_input_mode)

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

    input_stream2 = pars.get("input_stream2")
    if input_stream2:
        bsread_address2 = input_stream2
        bsread_channels2 = None
    else:
        bsread_address2 = pars.get("bsread_address2")
        bsread_channels2 = pars.get("bsread_channels2")

    # Stream merging
    if bsread_address2 or bsread_channels2:
        merge_queue_size = pars.get("merge_queue_size", 100)
        merge_buffer_size = pars.get("merge_buffer_size", 100)
        bsread_mode2 = pars.get("input_mode2", default_input_mode)
        _logger.debug("Connecting to second stream %s. %s" % (str(bsread_address2), str(bsread_channels2)))
        if bsread_address2:
            bsread_mode2 = SUB if bsread_mode2 == "SUB" else PULL
        else:
            bsread_mode2 = PULL if bsread_mode2 == "PULL" else SUB

        if bsread_channels2 is not None:
            if type(bsread_channels2) != list:
                bsread_channels2 = json.loads(bsread_channels2)
            if len(bsread_channels2) == 0:
                bsread_channels2 = None
        merge_receive_timeout = int(pars.get("receive_timeout", 10))
        if bsread_address:
            st1 = merger.StreamSource(bsread_address, bsread_mode, merge_queue_size, merge_receive_timeout)
        else:
            st1 = merger.DispatcherSource(bsread_channels, dispatcher_url, dispatcher_verify_request, dispatcher_disable_compression, merge_queue_size, merge_receive_timeout)
        if bsread_address2:
            st2 = merger.StreamSource(bsread_address2, bsread_mode2, merge_queue_size, merge_receive_timeout)
        else:
            st2 = merger.DispatcherSource(bsread_channels2, dispatcher_url, dispatcher_verify_request, dispatcher_disable_compression, merge_queue_size, merge_receive_timeout)

        ret = merger.Merger(st1, st2, receive_timeout, merge_buffer_size)
        ret.connect()
        source = ret
        return ret

    ret = bssource(  host=bsread_host,
                      conn_type=conn_type,
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


def send_data(processed_data, global_timestamp, pulse_id):
    global pid_buffer, pid_buffer_size
    if pid_buffer is not None:
        pid_buffer[pulse_id] = (processed_data, global_timestamp, pulse_id)
        while len(pid_buffer) >= pid_buffer_size:
            pulse_id = min(pid_buffer.keys())
            tx = pid_buffer.pop(pulse_id)
            _send_data(*tx, _message_buffer)
    else:
        return _send_data(processed_data, global_timestamp, pulse_id, _message_buffer)

def _send_data(processed_data, global_timestamp, pulse_id, message_buffer = None):
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

        if message_buffer is not None:
            message_buffer.append((processed_data, global_timestamp, pulse_id))
        else:
            send(sender, processed_data, global_timestamp, pulse_id)
        #_logger.debug("Sent PID %d" % (pulse_id,))


def receive_stream(camera=False):
    global last_rcvd_timestamp, pause, pid_range, downsampling, downsampling_counter, camera_timeout, stream_timeout, stream_failed
    pulse_id = global_timestamp = data = None
    rx = source.receive()

    if rx:
        stream_failed = False
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
                        _logger.warning("Unexpected PID: " + str(pulse_id) + " -  last: " + str(former_pid) + ", cur:" + str(current_pid) + ", lost:" + str(lost) + "." + get_log_tag())
                    else:
                        _logger.debug("Newer PID: " + str(pulse_id) + " -  last: " + str(former_pid) + ", cur:" + str(current_pid) + "." + get_log_tag())
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
                _logger.warning("Reached end of pid range: stopping pipeline.%s" % get_log_tag())
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
        stream_failed = True
        if abort_on_timeout():
            if (stream_timeout > 0) and (time.time() - last_rcvd_timestamp) > stream_timeout:
                _logger.warning("Stream timeout.%s" % get_log_tag())
                raise SourceTimeout("Stream Timeout")
        if camera:
            if camera_timeout:
                if (camera_timeout > 0) and (time.time() - last_rcvd_timestamp) > camera_timeout:
                    _logger.warning("Camera timeout.%s" % get_log_tag())
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
    global number_processing_threads, received_pids, tx_lock, thread_buffers, message_buffer, message_buffer_size, max_frame_rate,  last_sent_timestamp

    # Check maximum frame rate parameter
    if max_frame_rate:
        min_interval = 1.0 / max_frame_rate
        if (time.time() - last_sent_timestamp) < min_interval:
            return
    try:
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
                        _logger.error("Thread %d buffer full: lost PID %d.%s" % (index, lost_pid, get_log_tag()))
            return
        processed_data = _process_data(processing_function, pulse_id, global_timestamp, function, *args)
        if processed_data is not None:
            _send_data(processed_data, global_timestamp, pulse_id, message_buffer)
    finally:
        last_sent_timestamp = time.time()


def _process_data(processing_function, pulse_id, global_timestamp, function, *args):
    if config.TELEMETRY_ENABLED:
        global pipeline_name, camera_name, output_stream_port
        with tracer.start_as_current_span("process") as process_span:
            process_span.set_attribute("pipeline", pipeline_name)
            process_span.set_attribute("camera", camera_name)
            process_span.set_attribute("port", output_stream_port)
            process_span.set_attribute("pulse_id", pulse_id)
            process_span.set_attribute("thread_id", threading.get_ident())
            process_span.set_attribute("timestamp", global_timestamp)
            process_span.set_attribute("function", str(function.__name__))
            start = time.time()
            try:
                processed_data = processing_function(pulse_id, global_timestamp, function, *args)
                process_span.set_attribute("ex", "")
                run_counter.add(1, {"success": True})
                run_timespan.record(time.time()-start, {"success": True})
                return processed_data
            except Exception as e:
                process_span.set_attribute("ex", str(e))
                process_span.set_status(Status(StatusCode.ERROR))
                run_counter.add(1, {"success": False})
                run_timespan.record(time.time()-start, {"success": False})
                raise
    else:
        return processing_function(pulse_id, global_timestamp, function, *args)


def setup_sender(output_port, stop_event, pipeline_processing_function=None, user_scripts_manager=None):
    global number_processing_threads, multiprocessed, processing_thread_index, received_pids, processing_threads, message_buffer_send_thread, tx_lock, thread_buffers, message_buffer_size, _message_buffer
    pars = get_parameters()
    number_processing_threads = pars.get("processing_threads", 0)
    thread_buffers = None if number_processing_threads == 0 else []
    multiprocessed = pars.get("multiprocessing", False)

    message_buffer_size = pars.get("buffer_size")
    if message_buffer_size:
        message_buffer = deque(maxlen=message_buffer_size)
        message_buffer_send_thread = Thread(target=message_buffer_send_task, args=(message_buffer, output_port, stop_event))
        message_buffer_send_thread.start()
        _message_buffer = message_buffer

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
                message_buffer_send_thread = multiprocessing.Process(target=process_send_task, args=(output_port, tx_queue, received_pids, spawn_send_thread, stop_event, pars, get_log_suffix()))
                message_buffer_send_thread.start()
                for i in range(number_processing_threads):
                    thread_buffer = multiprocessing.Queue(thread_buffer_size)
                    thread_buffers.append(thread_buffer)
                    processing_thread = multiprocessing.Process(target=process_task, args=( #TODO change config in processes
                    pipeline_processing_function, thread_buffer, tx_queue, stop_event, i, user_scripts_manager, pars, get_log_suffix()))
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
    global sender
    pars = get_parameters()
    _logger.info("Start message buffer send thread.%s" % get_log_tag())
    create_sender(output_port, stop_event)
    try:
        while not stop_event.is_set():
            if len(message_buffer) == 0:
                time.sleep(0.001)
            else:
                (processed_data, timestamp, pulse_id) = message_buffer.popleft()
                send(sender, processed_data, timestamp, pulse_id)

    except Exception as e:
        _logger.error("Error on message buffer send thread: %s.%s" % (str(e), get_log_tag()))
    finally:
        stop_event.set()
        if sender:
            try:
                sender.close()
            except:
                pass
            finally:
                sender = None
        _logger.info("Exit message buffer send thread.%s" % get_log_tag())


#Multi-threading
def thread_task(process_function, thread_buffer, tx_buffer, tx_lock, received_pids, stop_event, index):
    global thread_exit_code, debug
    _logger.info("Start processing thread %d: %d.%s" % (index,threading.get_ident(), get_log_tag()))
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
            processed_data = _process_data(process_function, pulse_id, global_timestamp, function, *args)
            if processed_data is None:
                if debug:
                    _logger.info ("Error processing PID %d at thread %d.%s" % (pulse_id, index, get_log_tag()))
                try:
                    with tx_lock:
                        received_pids.remove(pulse_id)
                except:
                    _logger.warning("Error removing PID %d at thread %d.%s" % (pulse_id, index, get_log_tag()))
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
                        _logger.info("Send buffer full - removing oldest PID: %d at thread %d.%s" % (pulse_id, index, get_log_tag()))

    except Exception as e:
        thread_exit_code = 2
        _logger.error("Error on processing thread %d:%s" % (index, str(e), get_log_tag() ))
    finally:
        stop_event.set()
        _logger.info("Exit processing thread %d.%s" % (index, get_log_tag()))


def thread_send_task(output_port, tx_buffer, tx_lock, received_pids, stop_event):
    _logger.info("Start threaded processing send thread.%s" % get_log_tag())
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
                _send_data(*tx)
            else:
                if popped:
                    if debug:
                        _logger.error("Removed timed-out processing -  Pulse ID: %s.%s" % (str(pid),  get_log_tag()))
                time.sleep(0.01)
    except Exception as e:
        _logger.error("Error on threaded processing send thread: %s.%s" % (str(e), get_log_tag()))
    finally:
        stop_event.set()
        if sender:
            try:
                sender.close()
            except:
                pass
        _logger.info("Exit threaded processing send thread.%s" % get_log_tag())


# Multi-processing
def process_task(process_function, thread_buffer, tx_queue, stop_event, index, user_scripts_manager, pipeline_config, ltag):
    #global log_tag
    global _parameters
    #log_tag = ltag
    set_log_suffix(ltag)
    _logger.info("Start process %d: %d.%s" % (index, os.getpid(), get_log_tag()))

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
            processed_data = _process_data(process_function, pulse_id, global_timestamp, function, *args)
            try:
                tx_queue.put((processed_data, global_timestamp, pulse_id, message_buffer), False)
            except Exception as e:
                _logger.error("Error adding to tx buffer %d: %s.%s" % (index, str(e), get_log_tag()))
            #_logger.info("Processed message %d in process %d" % (pulse_id, index))


    except Exception as e:
        _logger.error("Error on process %d: %s.%s" % (index, str(e), get_log_tag()))
        #import traceback
        #traceback.print_exc()
    finally:
        stop_event.set()
        _logger.info("Exit process  %d.%s" % (index, get_log_tag()))
        sys.exit(0)


def process_send_task(output_port, tx_queue, received_pids_queue, spawn_send_thread, stop_event, pars, ltag):
    global log_tag
    log_tag = ltag

    sender = None
    _parameters = pars
    _logger.info("Start send process.%s" % get_log_tag())
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
                                    _logger.error("Timeout processing PID %s.%s" % (str(pid), get_log_tag()))
                        if tx is not None:
                            _send_data(processed_data, global_timestamp, pulse_id, message_buffer)
                        else:
                            break
                # When multiprocessed cannot access sender object from main process
                if ((time.time() - last_msg_timestamp) > 1.0):
                    get_statistics().num_clients = get_clients(get_sender())
                    last_msg_timestamp = time.time()

    except Exception as e:
        _logger.error("Error send process %s.%s" % (str(e), get_log_tag()))
    finally:
        stop_event.set()
        if sender:
            try:
                sender.close()
            except:
                pass
        _logger.info("Exit send process.%s" % get_log_tag())
        sys.exit(0)


def get_lib_file(name):
    return os.path.abspath(_user_scripts_manager.get_lib_path(name))


def import_egg(egg_name):
    egg_path = get_lib_file(egg_name)
    if (egg_path):
        import pkg_resources
        sys.path.append(egg_path)
        pkg_resources.working_set.add_entry(egg_path)


def cleanup(exit_code=0):
    _logger.info("Stopping transceiver.%s" % get_log_tag())
    global source, sender

    if source:
        try:
            if stream_failed:
                _logger.info("Source timeout.%s" % get_log_tag())
            else:
                _logger.info("Disconnecting source.%s" % get_log_tag())
            source.disconnect()
        except:
            pass
        finally:
            _logger.info("Source disconnected.%s" % get_log_tag())
            source = None

    if message_buffer_send_thread:
        try:
            _logger.info("Joining message buffer send thread.%s" % get_log_tag())
            message_buffer_send_thread.join(0.1)
        except:
            pass
        finally:
            _logger.info("Joined message buffer send thread.%s" % get_log_tag())
    else:
        if sender:
            try:
                _logger.info("Closing sender.%s" % get_log_tag())
                sender.close()
            except:
                pass
            finally:
                _logger.info("Closed sender.%s" % get_log_tag())
                sender = None
    i = 1
    for t in processing_threads:
        if t:
            try:
                _logger.info("Joining processing thread %s.%s" % (str(i), get_log_tag()))
                i = i+1
                t.join(0.1)
            except:
                pass
            finally:
                _logger.info("Joined processing thread %s.%s" % (str(i), get_log_tag()))
    if thread_exit_code != 0:
        exit_code = thread_exit_code
    _logger.info("Exiting process with exit_code %d.%s" % (exit_code, get_log_tag()))
    #sys.exit sometimes leave defunct process, so the pipeline  cannot restart
    os.kill(os.getpid(), signal.SIGTERM)
    _logger.warning("Passed by kill %s" % get_log_tag())
    sys.exit(exit_code)
    _logger.warning("Passed by exit.%s" % get_log_tag())

