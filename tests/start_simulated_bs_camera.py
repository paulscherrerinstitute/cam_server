
import os
import time
import json
from threading import Thread

import multiprocessing

from bsread import SUB, source
from os import listdir

from mflow import mflow, PUSH, sleep
from os.path import join, isfile

stop_event = multiprocessing.Event()
simulate_pid = True


def tx_task(bind_address, input_folder, stop_event):
    files = sorted(listdir(input_folder))
    stream = mflow.connect(bind_address, conn_type="bind", mode=PUSH)
    pid = 3682521968
    try:
        while not stop_event.is_set():

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
                            h["pulse_id"] = pid
                            time.sleep(0.017)
                            pid = pid + 1
                            data = bytes(json.dumps(h), 'utf-8')
                            #print("")
                        #print('Sending %s [%s]' % (raw_file, send_more))
                        #print (pid , end=" ")
                        stream.send(data, send_more=send_more)
    finally:
        mflow.disconnect()

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