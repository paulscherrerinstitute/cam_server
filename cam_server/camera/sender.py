import sys
import time
from logging import getLogger
from threading import Thread, RLock
from collections import deque
import numpy

from zmq import Again

from cam_server import config
from cam_server.camera.source.common import transform_image
from cam_server.otel import otel_get_meter, otel_setup_logs
from cam_server.utils import update_statistics, on_message_sent, init_statistics, MaxLenDict, timestamp_as_float, \
    synchronise_threads, setup_instance_logs, set_log_suffix

_logger = getLogger(__name__)

if config.TELEMETRY_ENABLED:
    otel_setup_logs()
    meter = otel_get_meter()
    rx_counter = meter.create_counter("rx_counter", description="Frame receive counter")
    error_counter = meter.create_counter("rx_counter", description="Frame receive counter")

def process_epics_camera(stop_event, statistics, parameter_queue, logs_queue, camera, port):
    """
    Start the camera stream and listen for image monitors. This function blocks until stop_event is set.
    :param stop_event: Event when to stop the process.
    :param statistics: Statistics namespace.
    :param parameter_queue: Parameters queue to be passed to the pipeline.
    :param camera: Camera instance to get the images from.
    :param port: Port to use to bind the output stream.
    """
    sender = None
    exit_code = 0
    try:
        set_log_suffix(" [camera:%s]" % camera.get_name())
        setup_instance_logs(logs_queue)
        init_statistics(statistics)
        def process_parameters():
            nonlocal x_size, y_size, x_axis, y_axis
            x_size, y_size = shape = camera.get_geometry()
            x_axis, y_axis = camera.get_x_y_axis()
            dtype = camera.get_dtype()
            camera.register_channels_change_type_shape(dtype, shape)

        x_size = y_size = x_axis = y_axis = None
        camera.connect()
        sender = camera.create_sender(stop_event, port)

        # Register the bsread channels - compress only the image.
        camera.register_channels(False)

        process_parameters()

        def collect_and_send(image, timestamp, shape_changed = False):
            if config.TELEMETRY_ENABLED:
                rx_counter.add(1)

            nonlocal x_size, y_size, x_axis, y_axis

            if shape_changed:
                process_parameters()

            data = {"image": image,
                    "timestamp": timestamp,
                    "width": x_size,
                    "height": y_size,
                    "x_axis": x_axis,
                    "y_axis": y_axis}
            frame_size = ((image.size * image.itemsize) if (image is not None) else 0)
            frame_shape = str(x_size) + "x" + str(y_size) + "x" + str(image.itemsize)
            update_statistics(sender, -frame_size, 1 if (image is not None) else 0, frame_shape, camera.forwarder)

            try:
                pulse_id = camera.get_pulse_id()
                camera.send(data, pulse_id, timestamp)
            except Again:
                _logger.warning("Send timeout. Lost image with timestamp '%s'." % (str(timestamp)))

            while not parameter_queue.empty():
                new_parameters = parameter_queue.get()
                camera.camera_config.set_configuration(new_parameters)
                process_parameters()

        camera.add_callback(collect_and_send)

        # This signals that the camera has successfully started.
        stop_event.clear()
        _logger.info("Camera started")

    except:
        _logger.exception("Error while processing camera stream.")
        exit_code = 1

    finally:

        # Wait for termination / update configuration / etc.
        stop_event.wait()
        camera.disconnect()
        camera.close_sender()
        sys.exit(exit_code)



def process_bsread_camera(stop_event, statistics, parameter_queue, logs_queue, camera, port):
    """
    Start the camera stream and receive the incoming bsread streams. This function blocks until stop_event is set.
    :param stop_event: Event when to stop the process.
    :param statistics: Statistics namespace.
    :param parameter_queue: Parameters queue to be passed to the pipeline.
    :param camera: Camera instance to get the stream from.
    :param port: Port to use to bind the output stream.
    """
    sender = None
    stream_lock = RLock()
    camera_streams = []
    receive_threads = []
    threaded = False
    buffered = False
    message_buffer, message_buffer_send_thread, message_buffer_lock = None, None, None
    data_changed = False
    format_error = False
    exit_code = 0
    fail_counter = 0
    enforce_pid = True
    enforce_timestamp = True
    check_timestamp = True
    try:
        set_log_suffix(" [camera:%s]" % camera.get_name())
        setup_instance_logs(logs_queue)
        init_statistics(statistics)


        def process_parameters():
            nonlocal x_size, y_size, x_axis, y_axis, enforce_pid, enforce_timestamp, check_timestamp
            x_axis, y_axis = camera.get_x_y_axis()
            x_size, y_size = camera.get_geometry()
            enforce_pid = camera.camera_config.parameters.get("enforce_pid", True)
            enforce_timestamp = camera.camera_config.parameters.get("enforce_pid", True)
            check_timestamp = camera.camera_config.parameters.get("check_timestamp", True)

        def data_change_callback(channels):
            nonlocal data_changed
            data_changed = True

        def send_task(message_buffer, stop_event):
            nonlocal sender, exit_code
            _logger.info("Start send thread.")
            sender = camera.create_sender(stop_event, port, True)
            debug = camera.get_debug()
            try:
                while not stop_event.is_set():
                    if len(message_buffer) == 0:
                        time.sleep(0.001)
                    else:
                        (data, pulse_id, timestamp) = message_buffer.popleft()
                        camera.send(data, pulse_id, timestamp)
                _logger.info("stop_event set to send thread.")
            except Exception as e:
                exit_code = 2
                _logger.exception("Error on send thread: %s." % (str(e)))
            finally:
                stop_event.set()
                camera.close_sender()
                _logger.info("Exit send thread.")

        def message_buffer_send_task(message_buffer, connections, stop_event, message_buffer_lock):
            nonlocal sender, exit_code
            _logger.info("Start message buffer send thread.")
            sender = camera.create_sender(stop_event, port, True)
            message_buffer.last_pid = -1
            last_tx_pid = -1
            interval = 1
            threshold = int(message_buffer.maxlen * camera.get_buffer_threshold())
            debug = camera.get_debug()
            try:
                while not stop_event.is_set():
                    tx = None
                    old_pid=None
                    with message_buffer_lock:
                        size=len(message_buffer)
                        if size > 0:
                            pids = sorted(message_buffer.keys())
                            pulse_id = pids[0]
                            if connections==1:
                                message_buffer.last_pid = pulse_id
                                tx = message_buffer.pop(pulse_id)
                            else:
                                if pulse_id <= last_tx_pid:
                                    message_buffer.pop(pulse_id)  # Remove ancient PIDs
                                    old_pid = pulse_id
                                else:
                                    if pulse_id <= (last_tx_pid+interval) or (size >= threshold):
                                        message_buffer.last_pid = pulse_id
                                        tx = message_buffer.pop(pulse_id)
                    if tx is not None:
                        (data, timestamp) = tx
                        camera.send(data, pulse_id, timestamp)
                        if last_tx_pid>0:
                            expected = (last_tx_pid + interval)
                            if pulse_id != expected:
                                interval = pulse_id - last_tx_pid
                                if debug:
                                    if pulse_id > expected:
                                        _logger.info ("Failed Pulse ID:  expecting %d - received %d: Pulse ID interval set to: %d." % (expected, pulse_id, interval))
                                    else:
                                        _logger.info ("Changed interval: expecting %d - received %d: Pulse ID interval set to: %d." % (expected, pulse_id, interval))
                        last_tx_pid = pulse_id
                    elif old_pid is not None:
                        if debug:
                            _logger.info("Removed ancient Pulse ID from queue: %d  - Last: %d." % (pulse_id, last_tx_pid))
                    if size == 0:
                        time.sleep(0.001)
                    #while not parameter_queue.empty():
                    #    new_parameters = parameter_queue.get()
                    #    camera.camera_config.set_configuration(new_parameters)
                    #    process_parameters()

                _logger.info("stop_event set to send thread.")
            except Exception as e:
                exit_code = 2
                _logger.exception("Error on message buffer send thread: %s.")
            finally:
                stop_event.set()
                camera.close_sender()
                _logger.info("Exit message buffer send thread.")


        def flush_stream(camera_stream):
                while camera_stream.stream.receive(handler=camera_stream.handler.receive, block=False) is not None:
                    pass

        def is_interleaved(r1, r2):
            return (r1[0] < r2[0] < r1[1] < r2[1]) or (r2[0] < r1[0] < r2[1] < r1[1])

        def is_near(r1, r2):
            return abs(r2[1] - r1[1]) <= (buffer_size / connections / 2)

        def is_regular(r1, r2):
            return (r1[1] - r1[0]) == (r2[1] - r2[0])

        pid_ranges = None

        def get_pid_ranges():
            nonlocal camera_streams, stream_lock, pid_ranges
            with stream_lock:
                last_pids = [camera_stream.last_pulse_ids for camera_stream in camera_streams]
                pid_ranges = [(min(pids), max(pids)) if (len(pids) == pids.maxlen) else None for pids in last_pids]
            return pid_ranges

        def are_streams_aligned():
            nonlocal connections
            if connections > 1:
                pid_ranges = get_pid_ranges()
                if not None in pid_ranges:
                    for i in range(1, len(pid_ranges)):
                        if not is_regular(pid_ranges[i - 1], pid_ranges[i]) or (
                                not is_near(pid_ranges[i - 1], pid_ranges[i]) and not is_interleaved(pid_ranges[i - 1], pid_ranges[i])):
                            return False
            return True

        def assert_streams_aligned(max_fail_count=0):
            nonlocal fail_counter, pid_ranges, connections
            if connections > 1:
                if not are_streams_aligned():
                    if fail_counter == 0:
                        if camera.get_debug():
                            _logger.info("Misalignment - counter:%d %s." % (fail_counter, str(pid_ranges)))
                    fail_counter += 1
                else:
                    if fail_counter > 0:
                        if camera.get_debug():
                            _logger.info("Aligned      - counter:%d %s." % (fail_counter, str(pid_ranges)))
                    fail_counter = 0
                # If streams keep misaligned for 30s the restart them
                if fail_counter > max_fail_count:
                    raise Exception("Streams misaligned - aborting")


        x_size = y_size = x_axis = y_axis = None

        camera.check_data = True
        camera.connect()
        camera_name = camera.get_name()

        connections = camera.get_connections()
        buffer_size = camera.get_buffer_size()
        buffered = buffer_size > 0
        threaded = camera.is_threaded()

        process_parameters()
        # register_image_channel(x_size, y_size, dtype)

        # This signals that the camera has successfully started.
        stop_event.clear()
        _logger.info("Camera started")

        stats_lock = RLock()
        if buffered:
            message_buffer_lock = RLock()
            message_buffer = MaxLenDict(maxlen=buffer_size)
            message_buffer_send_thread = Thread(target=message_buffer_send_task, args=(message_buffer, connections, stop_event, message_buffer_lock))
            message_buffer_send_thread.start()
        elif threaded:
            message_buffer=deque(maxlen=10)
            message_buffer_send_thread = Thread(target=send_task, args=(message_buffer, stop_event))
            message_buffer_send_thread.start()
        else:
            sender = camera.create_sender(stop_event, port, True)


        if connections > 1:
            _logger.info("Connecting to camera over bsread with %d connections (buffer size = %d)" % (connections, buffer_size))
        else:
            _logger.info("Connecting to camera over bsread (buffer size = %d)" % ( buffer_size))


        if not buffered:
            for i in range(connections):
                stream = camera.get_stream(data_change_callback=data_change_callback)
                #stream.format_error_counter = 0
                camera_streams.append(stream)
                stream.connect()

        last_tx_pid = None
        total_bytes = [0] * connections
        frame_shape = None

        def process_stream(camera_stream, index):
            nonlocal total_bytes, last_tx_pid, frame_shape, format_error, fail_counter
            try:
                if stop_event.is_set():
                    return
                data = camera_stream.receive()

                if data is not None:
                    pulse_id = data.data.pulse_id
                    timestamp = (data.data.global_timestamp, data.data.global_timestamp_offset)
                    image = data.data.data[camera_name + config.EPICS_PV_SUFFIX_IMAGE].value
                    if image is None:
                        if camera.get_debug():
                            _logger.info("Format error: no image.")
                        format_error = True #on_format_error()
                        return pulse_id, timestamp, None
                    else:
                        if config.TELEMETRY_ENABLED:
                            rx_counter.add(1)
                        # Rotate and mirror the image if needed - this is done in the epics:_get_image for epics cameras.
                        image = transform_image(image, camera.camera_config)

                        # Numpy is slowest dimension first, but bsread is fastest dimension first.
                        height, width = image.shape
                        if (len(x_axis)!=width) or (len(y_axis)!=height):
                            format_error = True #on_format_error()
                            if camera.get_debug():
                                _logger.info("Format error - bad axis size.")
                            return pulse_id, timestamp, None
                        format_error = False
                        frame_shape = str(width) + "x" + str(height) + "x" + str(image.itemsize)
                        total_bytes[index] = data.statistics.total_bytes_received

                with stats_lock:
                    update_statistics(sender, sum(total_bytes), 1 if data else 0, frame_shape, camera.forwarder)

                # In case of receiving error or timeout, the returned data is None.
                if data is None:
                    if buffered and camera.get_debug():
                        _logger.info("Null data.")
                    return None


                with stream_lock:
                    camera_stream.last_pulse_ids.append(pulse_id)

                data = {
                    "image": image,
                    "height": height,
                    "width": width,
                    "x_axis": x_axis,
                    "y_axis": y_axis,
                    "timestamp": timestamp_as_float(timestamp)
                }
                if buffered:
                    with message_buffer_lock:
                        last_pid = message_buffer.last_pid
                        if pulse_id > last_pid:
                            message_buffer[pulse_id] = (data, timestamp)
                    if pulse_id <= last_pid:
                        if camera.get_debug():
                            _logger.warning("Invalid pulse id on stream %d: %d - Last: %d." % (index, pulse_id, last_pid))
                        #flush_stream(camera_stream)
                    #else:
                    #    _logger.info("Put : %d." % (pulse_id,))
                else:
                    return pulse_id, timestamp, data
            except Exception as e:
                raise

        def receive_task(index, message_buffer, stop_event, message_buffer_lock, camera_stream, connections):
            global exit_code
            _logger.info("Start receive thread %d." % (index, ))
            camera_stream.connect()
            if connections>1:
                synchronise_threads(connections)
                flush_stream(camera_stream)

            try:
                while not stop_event.is_set():
                    process_stream(camera_stream, index)
                _logger.info("stop_event set to receive thread %d." % (index))
            except Exception as e:
                _logger.exception("Error on receive thread %d: %s." % (index, str(e)))
                exit_code = 4
            finally:
                _logger.info("Exiting receive thread %d." % (index))
                stop_event.set()
                if camera_stream:
                    try:
                        camera_stream.disconnect()
                    except:
                        pass
                _logger.info("Exit receive thread %d." % (index,))

        if buffered:
            for i in range(connections):
                camera_stream = camera.get_stream(data_change_callback=data_change_callback)
                receive_thread = Thread(target=receive_task, args=(i, message_buffer, stop_event, message_buffer_lock, camera_stream, connections))
                receive_threads.append(receive_thread)
                camera_streams.append(camera_stream)

            for receive_thread in receive_threads:
                receive_thread.start()

        start_error = 0
        stream_buffers = [None,] * len(camera_streams)
        last_pid = -1
        last_timestamp_float = -1
        last_timestamp = None
        while not stop_event.is_set():
            while not parameter_queue.empty():
                new_parameters = parameter_queue.get()
                camera.camera_config.set_configuration(new_parameters)
                process_parameters()

            if data_changed:
                time.sleep(0.1) #Sleeping in case channels are monitored and were not updated
                camera.update_size_raw()
                process_parameters()
                _logger.warning("Image shape changed: %dx%d." % (x_size, y_size))
                if buffered:
                    time.sleep(0.25) #If buffered give some time to other threads report the change
                data_changed = False

            if format_error:
                now=time.time()
                if start_error<=0:
                    _logger.warning("Invalid image format.")
                    start_error = now
                else:
                    if (now-start_error) >config.BSREAD_FORMAT_ERROR_TIMEOUT:
                        _logger.error("Invalid image format timeout: stopping instance.")
                        stop_event.set()
                        exit_code = 5
                        break
            else:
                if start_error > 0:
                    _logger.info("Image format ok.")
                start_error = 0

            if buffered:
                if camera.abort_on_error():
                    assert_streams_aligned(1000)
                time.sleep(0.01)
            else:
                for i in range(len(camera_streams)):
                    if stream_buffers[i] is None:
                        stream_buffers[i] = process_stream(camera_streams[i], i)
                #pids = [(sb[1] if (sb is not None) else sys.maxsize) for sb in stream_buffers]
                #pid = min(pids)
                #if pid < sys.maxsize:

                if None not in stream_buffers:
                    pids = [sb[0] for sb in stream_buffers]
                    pid = min(pids)
                    i = pids.index(pid)
                    pulse_id, timestamp, data = stream_buffers[i]
                    stream_buffers[i] = None

                    if (pulse_id is not None) and (timestamp is not None):
                        if (not check_timestamp) or ((pulse_id % 1000000) == (timestamp[1] % 1000000)):
                            if (pulse_id > last_pid) or (not enforce_pid):
                                timestamp_float = timestamp_as_float(timestamp)
                                if (timestamp_float > last_timestamp_float) or (not enforce_timestamp):
                                        if data is not None:
                                            if threaded:
                                                message_buffer.append((data, pulse_id, timestamp))
                                            else:
                                                camera.send(data, pulse_id, timestamp)
                                        last_pid = pulse_id
                                        last_timestamp = timestamp
                                        last_timestamp_float = timestamp_float
                                else:
                                    if camera.get_debug():
                                        _logger.warning("Received invalid Timestamp: %s %f - last: %s %f PID: %d - last: %d." % (str(timestamp), timestamp_float, str(last_timestamp),last_timestamp_float, pulse_id, last_pid))
                            else:
                                if camera.get_debug():
                                    _logger.info("Received old Pulse ID on stream %d: %d - last: %d.]" % (i, pulse_id, last_pid))
                                flush_stream(camera_streams[i])
                        else:
                            if camera.get_debug():
                                _logger.info("Received incompatible Timestamp: %s  PID: %d." % (str(timestamp), pulse_id))
    except Exception as e:
        _logger.exception("Error while processing camera stream: %s." % (str(e)))
        exit_code = 1

    finally:
        _logger.info("Stopping sender.")
        stop_event.set()

        if camera:
            try:
                camera.disconnect()
            except:
                pass

        if not buffered:
            for stream in camera_streams:
                try:
                    stream.disconnect()
                except:
                    pass
            if not threaded:
                camera.close_sender()
        else:
            for t in receive_threads + [message_buffer_send_thread]:
                if t:
                    try:
                        t.join(0.1)
                    except:
                        pass
        _logger.info("Exiting process.")
        sys.exit(exit_code)


def process_standard_camera(stop_event, statistics, parameter_queue, logs_queue, camera, port):
    camera.process(stop_event, statistics, parameter_queue, logs_queue, port)


source_type_to_sender_function_mapping = {
    "epics": process_epics_camera,
    "simulation": process_epics_camera,
    "area_detector": process_epics_camera,
    "bsread": process_bsread_camera,
    "bsread_simulation": process_bsread_camera,
    "stream": process_standard_camera,
    "hdf5": process_standard_camera,
    "array10": process_standard_camera,
    "script": process_standard_camera
}


def get_sender_function(source_type_name):
    if source_type_name not in source_type_to_sender_function_mapping:
        raise ValueError("source_type '%s' not present in sender function pv_to_stream_mapping. Available: %s." %
                         (source_type_name, list(source_type_to_sender_function_mapping.keys())))

    return source_type_to_sender_function_mapping[source_type_name]