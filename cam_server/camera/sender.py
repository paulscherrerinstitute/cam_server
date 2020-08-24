import time
from logging import getLogger

from bsread.sender import Sender, PUB
from zmq import Again

from cam_server import config
from cam_server.camera.source.common import transform_image
from cam_server.utils import set_statistics, init_statistics, MaxLenDict

from threading import Thread, RLock
#from epics.ca import CAThread

_logger = getLogger(__name__)


def get_client_timeout(camera):
    client_timeout = camera.camera_config.get_configuration().get("no_client_timeout")
    if client_timeout is not None:
        return client_timeout
    return config.MFLOW_NO_CLIENTS_TIMEOUT

def get_connections(camera):
    connections = camera.camera_config.get_configuration().get("connections")
    try:
        if connections is not None:
            return max(int(connections), 1)
    except:
        _logger.warning("Invalid number of connections (using 1) [%s]" % (camera.get_name(),))
    return 1

def get_buffer_size(camera):
    buffer_size = camera.camera_config.get_configuration().get("buffer_size")
    try:
        if buffer_size is not None:
            return max(int(buffer_size), 0)
    except:
        _logger.warning("Invalid buffer size (using 0) [%s]" % (camera.get_name(),))
    return 0

def process_epics_camera(stop_event, statistics, parameter_queue, camera, port):
    """
    Start the camera stream and listen for image monitors. This function blocks until stop_event is set.
    :param stop_event: Event when to stop the process.
    :param statistics: Statistics namespace.
    :param parameter_queue: Parameters queue to be passed to the pipeline.
    :param camera: Camera instance to get the images from.
    :param port: Port to use to bind the output stream.
    """
    sender = None

    try:
        init_statistics(statistics)

        # If there is no client for some time, disconnect.
        def no_client_timeout():
            client_timeout = get_client_timeout(camera)
            if client_timeout > 0:
                _logger.info("No client connected to the stream for %d seconds. Closing instance. [%s]" %
                             (client_timeout, camera.get_name()))
                stop_event.set()

        def process_parameters():
            nonlocal x_size, y_size, x_axis, y_axis, simulate_pulse_id
            x_size, y_size = camera.get_geometry()
            x_axis, y_axis = camera.get_x_y_axis()
            simulate_pulse_id=camera.camera_config.get_configuration().get("simulate_pulse_id")
            sender.add_channel("image", metadata={"compression": config.CAMERA_BSREAD_IMAGE_COMPRESSION,
                                                  "shape": [x_size, y_size],
                                                  "type": "uint16"})

        x_size = y_size = x_axis = y_axis = simulate_pulse_id = None
        camera.connect()

        sender = Sender(port=port, mode=PUB,
                        data_header_compression=config.CAMERA_BSREAD_DATA_HEADER_COMPRESSION)

        # Register the bsread channels - compress only the image.
        sender.add_channel("width", metadata={"compression": config.CAMERA_BSREAD_SCALAR_COMPRESSION,
                                              "type": "int64"})

        sender.add_channel("height", metadata={"compression": config.CAMERA_BSREAD_SCALAR_COMPRESSION,
                                               "type": "int64"})

        sender.add_channel("timestamp", metadata={"compression": config.CAMERA_BSREAD_SCALAR_COMPRESSION,
                                                  "type": "float64"})

        sender.add_channel("x_axis", metadata={"compression": config.CAMERA_BSREAD_SCALAR_COMPRESSION,
                                               "type": "float32"})

        sender.add_channel("y_axis", metadata={"compression": config.CAMERA_BSREAD_SCALAR_COMPRESSION,
                                               "type": "float32"})

        sender.open(no_client_action=no_client_timeout, no_client_timeout=get_client_timeout(camera))

        process_parameters()

        def collect_and_send(image, timestamp):
            nonlocal x_size, y_size, x_axis, y_axis, simulate_pulse_id

            # Data to be sent over the stream.
            data = {"image": image,
                    "timestamp": timestamp,
                    "width": x_size,
                    "height": y_size,
                    "x_axis": x_axis,
                    "y_axis": y_axis}
            frame_size = ((image.size * image.itemsize) if (image is not None) else 0)
            frame_shape = str(x_size) + "x" + str(y_size) + "x" + str(image.itemsize)
            set_statistics(statistics, sender, statistics.total_bytes + frame_size, 1 if (image is not None) else 0, frame_shape)

            try:
                pulse_id = int(time.time() *100) if simulate_pulse_id else None
                sender.send(data=data, pulse_id = pulse_id, timestamp=timestamp, check_data=False)
            except Again:
                _logger.warning("Send timeout. Lost image with timestamp '%s' [%s]." % (str(timestamp), camera.get_name()))

            while not parameter_queue.empty():
                new_parameters = parameter_queue.get()
                camera.camera_config.set_configuration(new_parameters)

                process_parameters()

        camera.add_callback(collect_and_send)

        # This signals that the camera has successfully started.
        stop_event.clear()

    except:
        _logger.exception("Error while processing camera stream [%s]" % (camera.get_name(),))

    finally:

        # Wait for termination / update configuration / etc.
        stop_event.wait()

        camera.disconnect()

        if sender:
            try:
                sender.close()
            except:
                pass



def process_bsread_camera(stop_event, statistics, parameter_queue, camera, port):
    """
    Start the camera stream and receive the incoming bsread streams. This function blocks until stop_event is set.
    :param stop_event: Event when to stop the process.
    :param statistics: Statistics namespace.
    :param parameter_queue: Parameters queue to be passed to the pipeline.
    :param camera: Camera instance to get the stream from.
    :param port: Port to use to bind the output stream.
    """
    sender = None
    camera_streams = []
    threaded = False
    message_buffer, message_buffer_send_thread, message_buffer_lock = None, None, None

    try:
        init_statistics(statistics)

        # If there is no client for some time, disconnect.
        def no_client_timeout():
            client_timeout = get_client_timeout(camera)
            if client_timeout > 0:
                _logger.info("No client connected to the stream for %d seconds. Closing instance [%s]." %
                             (client_timeout, camera.get_name()))
                stop_event.set()

        def process_parameters():
            nonlocal x_size, y_size, x_axis, y_axis
            x_axis, y_axis = camera.get_x_y_axis()
            x_size, y_size = camera.get_geometry()

        def message_buffer_send_task(message_buffer, stop_event, message_buffer_lock):
            nonlocal sender
            _logger.info("Start message buffer send thread [%s]" % (camera.get_name(),))
            sender = Sender(port=port, mode=PUB,data_header_compression=config.CAMERA_BSREAD_DATA_HEADER_COMPRESSION)
            sender.open(no_client_action=no_client_timeout, no_client_timeout=get_client_timeout(camera))
            last_pid = None
            interval = 1
            threshold = (message_buffer.maxlen * 0.8)
            try:
                while not stop_event.is_set():
                    do_sleep=True
                    with message_buffer_lock:
                        size=len(message_buffer)
                        if size > 0:
                            pids = sorted(message_buffer.keys())
                            pulse_id = pids[0]
                            if (last_pid) and (pulse_id <= last_pid):
                                message_buffer.pop(pulse_id) #Remove ancient PIDs
                            else:
                                if not last_pid or \
                                     (pulse_id <= (last_pid+interval)) or (size > threshold):
                                    (data, timestamp) = message_buffer.pop(pulse_id)
                                    sender.send(data=data, pulse_id=pulse_id, timestamp=timestamp, check_data=True)
                                    if (last_pid):
                                        if pulse_id != (last_pid + interval):
                                            _logger.info("Failed Pulse ID %d [%s]" % ((last_pid + interval), camera.get_name(),))
                                        if interval != (pulse_id - last_pid):
                                            interval = pulse_id - last_pid
                                            _logger.info("Pulse ID interval set to: %d [%s]" % (interval, camera.get_name()))
                                    last_pid = pulse_id
                                    do_sleep = False
                    if do_sleep:
                        time.sleep(0.001)

                _logger.info("stop_event set to send thread [%s]" % (camera.get_name(),))
            except Exception as e:
                _logger.error("Error on message buffer send thread: %s [%s]" % (str(e), camera.get_name()))
            finally:
                isset = stop_event.is_set()
                stop_event.set()
                if sender:
                    try:
                        sender.close()
                    except:
                        pass
                _logger.info("Exit message buffer send thread [%s]" % (camera.get_name(),))

        def flush_stream(camera_stream):
                while camera_stream.stream.receive(handler=camera_stream.handler.receive, block=False) is not None:
                    pass


        # TODO: Use to register proper channels. But be aware that the size and dtype can change during the running.
        # def register_image_channel(size_x, size_y, dtype):
        #     sender.add_channel("image", metadata={"compression": config.CAMERA_BSREAD_IMAGE_COMPRESSION,
        #                                           "shape": [size_x, size_y],
        #                                           "type": dtype})

        x_size = y_size = x_axis = y_axis = None
        camera_name = camera.get_name()

        connections = get_connections(camera)
        buffer_size = get_buffer_size(camera)
        threaded = buffer_size > 0

        process_parameters()
        # register_image_channel(x_size, y_size, dtype)

        # This signals that the camera has successfully started.
        stop_event.clear()

        if threaded:
            message_buffer_lock = RLock()
            message_buffer = MaxLenDict(maxlen=buffer_size)
            message_buffer_send_thread = Thread(target=message_buffer_send_task, args=(message_buffer, stop_event, message_buffer_lock))
            message_buffer_send_thread.start()
        else:
            sender = Sender(port=port, mode=PUB,data_header_compression=config.CAMERA_BSREAD_DATA_HEADER_COMPRESSION)
            sender.open(no_client_action=no_client_timeout, no_client_timeout=get_client_timeout(camera))


        if connections > 1:
            _logger.info("Connecting to camera '%s' over bsread with %d connections (buffer size = %d)" % (camera_name, connections, buffer_size))
        else:
            _logger.info("Connecting to camera '%s' over bsread (buffer size = %d)" % (camera_name, buffer_size))


        if not threaded:

            for i in range(connections):
                stream = camera.get_stream()
                camera_streams.append(stream)
                stream.connect()

            # If multiple streams, ensure they are aligned
            if connections > 1:
                pid_offset = None
                def flush_streams():
                    for camera_stream in camera_streams:
                        flush_stream(camera_stream)

                def get_next_pids():
                    pids = []
                    for camera_stream in camera_streams:
                        data = camera_stream.receive()
                        if data is None:
                            data = camera_stream.receive()
                        if data is None:
                            raise Exception("Received no data from stream: " + str(camera_stream))
                        pids.append(data.data.pulse_id)
                    return pids

                def check_pids(pids):
                    nonlocal pid_offset
                    pid_offset = pids[1] - pids[0]
                    for i in range(1, len(pids)):
                        if (pids[i] - pids[i - 1]) != pid_offset:
                            return False
                    return True

                def align_streams():
                    nonlocal camera_streams
                    retries = 50;
                    for retry in range(retries):
                        _logger.info("Aligning streams: retry - %d [%s]" % (retry, camera.get_name()))
                        # First flush streams.
                        flush_streams()

                        # Get a message from streams
                        pids = get_next_pids()

                        # Arrange the streams according to the PID
                        indexes = sorted(range(len(pids)), key=pids.__getitem__)
                        camera_streams = [camera_streams[x] for x in indexes]
                        pids = [pids[x] for x in indexes]

                        # Check if the PID offsets are constant
                        if not check_pids(pids):
                            if retry >= (retries - 1):
                                raise Exception("PID offsets of streams are not constant: " + str(pids))
                            else:
                                _logger.info("PID offsets of streams are not constant - retrying: %s [%s]" % (str(pids), camera.get_name()))
                        else:
                            _logger.info("Aligned streams: %s [%s]" % (str(pids), camera.get_name()))
                            break;
                align_streams()

        last_pid = None
        total_bytes = [0] * connections

        def process_stream(camera_stream, index):
            nonlocal total_bytes, last_pid
            try:
                if stop_event.is_set():
                    return False
                data = camera_stream.receive()

                if data is not None:
                    total_bytes[index] = data.statistics.total_bytes_received
                set_statistics(statistics, sender, sum(total_bytes), 1 if data else 0)

                # In case of receiving error or timeout, the returned data is None.
                if data is None:
                    return True

                image = data.data.data[camera_name + config.EPICS_PV_SUFFIX_IMAGE].value

                # Rotate and mirror the image if needed - this is done in the epics:_get_image for epics cameras.
                image = transform_image(image, camera.camera_config)

                # Numpy is slowest dimension first, but bsread is fastest dimension first.
                height, width = image.shape

                pulse_id = data.data.pulse_id

                if not threaded:
                    if connections > 1:
                        if last_pid:
                            if pulse_id != (last_pid + pid_offset):
                                _logger.warning("Wrong pulse offset: realigning streams last=%d current=%d [%s]" % (last_pid, pulse_id, camera.get_name()))
                                align_streams()
                                last_pid = None
                                return False
                        last_pid = pulse_id

                timestamp_s = data.data.global_timestamp
                timestamp_ns = data.data.global_timestamp_offset
                timestamp = timestamp_s + (timestamp_ns / 1e9)

                data = {
                    "image": image,
                    "height": height,
                    "width": width,
                    "x_axis": x_axis,
                    "y_axis": y_axis,
                    "timestamp": timestamp
                }
                #print (pulse_id)
                if threaded:
                    with message_buffer_lock:
                        message_buffer[pulse_id]= (data, timestamp)
                else:
                    sender.send(data=data, pulse_id=pulse_id, timestamp=timestamp, check_data=True)

                while not parameter_queue.empty():
                    new_parameters = parameter_queue.get()
                    camera.camera_config.set_configuration(new_parameters)

                    process_parameters()

            except Exception as e:
                _logger.error("Could not process message: %s [%s]" % (str(e), camera.get_name()))
                stop_event.set()
            return True


        def receive_task(index, message_buffer, stop_event, message_buffer_lock, camera_stream):
            _logger.info("Start receive thread %d [%s]" % (index, camera.get_name()))
            #camera_stream = camera.get_stream()
            camera_stream.connect()
            try:
                while not stop_event.is_set():
                    process_stream(camera_stream, index)
                _logger.info("stop_event set to receive thread %d [%s]" % (index, camera.get_name()))
            except Exception as e:
                _logger.error("Error on receive thread %d: %s [%s]" % (index, str(e), camera.get_name()))
            finally:
                stop_event.set()
                if camera_stream:
                    try:
                        camera_stream.disconnect()
                    except:
                        pass
                _logger.info("Exit receive thread %d [%s]" % (index, camera.get_name()))


        receive_threads = []
        if threaded:
            for i in range(connections):
                camera_stream = camera.get_stream()
                receive_thread = Thread(target=receive_task, args=(i, message_buffer, stop_event, message_buffer_lock, camera_stream))
                receive_thread.start()
                receive_threads.append(receive_thread)

        while not stop_event.is_set():
            if threaded:
                time.sleep(0.01)
            else:
                for i in range(len(camera_streams)):
                    if not process_stream(camera_streams[i], i):
                        break

        _logger.info("Stopping transceiver [%s]" % (camera.get_name(),))

    except Exception as e:
        _logger.exception("Error while processing camera stream: %s [%s]" % (str(e), camera.get_name()))

    finally:
        # Wait for termination / update configuration / etc.
        stop_event.wait()

        if camera:
            try:
                camera.disconnect()
            except:
                pass

        if not threaded:
            for stream in camera_streams:
                try:
                    stream.disconnect()
                except:
                    pass
            if sender:
                try:
                    sender.close()
                except:
                    pass


source_type_to_sender_function_mapping = {
    "epics": process_epics_camera,
    "simulation": process_epics_camera,
    "bsread": process_bsread_camera,
    "bsread_simulation" : process_bsread_camera
}


def get_sender_function(source_type_name):
    if source_type_name not in source_type_to_sender_function_mapping:
        raise ValueError("source_type '%s' not present in sender function pv_to_stream_mapping. Available: %s." %
                         (source_type_name, list(source_type_to_sender_function_mapping.keys())))

    return source_type_to_sender_function_mapping[source_type_name]
