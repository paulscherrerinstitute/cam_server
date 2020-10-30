
import os
import math
import time
import json
import struct
from threading import Thread
from bsread.data.helpers import get_value_bytes
import hashlib
import epics
from cam_server import config


import multiprocessing

from bsread import SUB, source
from os import listdir

from mflow import mflow, PUSH, sleep
from os.path import join, isfile

stop_event = multiprocessing.Event()
simulate_pid = True

use_files = False
CAMERA_NAME = "SLG-LCAM-C102"

shape=None


def caget(channel_name, timeout=config.EPICS_TIMEOUT, as_string=False):
    channel = epics.PV(channel_name)
    try:
        ret = channel.get(timeout=timeout, as_string=as_string)
        if ret is None:
            print("Error getting channel %s" % (channel_name))
        return ret
    finally:
        channel.disconnect()

def tx_task(bind_address, input_folder, stop_event):
    files = sorted(listdir(input_folder))
    stream = mflow.connect(bind_address, conn_type="bind", mode=PUSH)
    pid = 3682521968
    counter = 0
    try:
        while not stop_event.is_set():
            counter = counter + 1
            if use_files == True:
                for index, raw_file in enumerate(files):
                    if not stop_event.is_set():
                        filename = join(input_folder, raw_file)
                        if not (raw_file.endswith('.raw') and isfile(filename)):
                            continue

                        with open(filename, mode='rb') as file_handle:
                            send_more = False
                            header = False
                            if index + 1 < len(files):  # Ensure that we don't run out of bounds
                                send_more = raw_file.split('_')[0] == files[index + 1].split('_')[0]
                                header = raw_file[7:10] == "000"
                            data = file_handle.read()
                            if simulate_pid and header:
                                h = json.loads(data)
                                pid = int(time.time() * 100)
                                h["pulse_id"] = pid
                                #time.sleep(0.017)
                                time.sleep(0.5)
                                pid = pid + 1
                                data = bytes(json.dumps(h), 'utf-8')
                                #print("")
                            #print('Sending %s [%s]' % (raw_file, send_more))
                            #print (pid , end=" ")
                            stream.send(data, send_more=send_more)
            else:
                global shape
                pid = int(time.time() * 100)
                width = caget(CAMERA_NAME+ ":WIDTH")
                height = caget(CAMERA_NAME + ":HEIGHT")
                #if shape is None:
                shape = [height, width]

                type = "uint16"
                def get_array(xsize,ysize, type):
                    import numpy
                    x = numpy.arange(0, xsize) + int(pid)
                    y = numpy.arange(0, ysize)
                    return (y.reshape(ysize, 1) * x.reshape(1, xsize)).astype(type, copy=False)

                main_header = {
                    'pulse_id': pid,
                    'global_timestamp': {"ns": 865393891, "sec": 1508856598},
                    'htype': "bsr_m-1.1",
                    'hash': "f161ed9dabb6d2369b70aa5d37db528f",
                }


                data_header = {"channels":  [
                                            {"compression": "none",
                                             "encoding": "little",
                                             "modulo": 1,
                                             "name": CAMERA_NAME + ":FPICTURE",
                                             "offset": 0,
                                             "shape": (shape[1], shape[0]),
                                             "type": type}
                                        ],
                            "htype": "bsr_d-1.1"}

                data_header_compression = None
                data_header_bytes = get_value_bytes(json.dumps(data_header), data_header_compression)
                main_header['hash'] = hashlib.md5(data_header_bytes).hexdigest()
                #shape = [110, 119]
                stream.send(json.dumps(main_header).encode('utf-8'), send_more=True)
                stream.send(json.dumps(data_header).encode('utf-8'), send_more=True)

                value = get_array(shape[1], shape[0], type)
                print (value.shape)
                stream.send(get_value_bytes(value, None,type), send_more=True)

                timestamp = time.time()
                current_timestamp_epoch = int(timestamp)
                current_timestamp_ns = int(math.modf(timestamp)[0] * 1e9)
                endianess = '<'
                stream.send(struct.pack(endianess + 'q',current_timestamp_epoch) +struct.pack(endianess + 'q',current_timestamp_ns), send_more=False)
                time.sleep(0.5)

    finally:
        mflow.disconnect(stream)

test_base_dir = os.path.split(os.path.abspath(__file__))[0]
thread = Thread(target=tx_task, args=("tcp://0.0.0.0:9999", os.path.join(test_base_dir,"test_camera_dump"), stop_event))
thread.start()


try:
    while(True):
        time.sleep(0.2)
finally:
    stop_event.set()
    thread.join()

print ("Stop")