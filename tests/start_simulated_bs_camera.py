
import os
import time
from threading import Thread

import multiprocessing

from bsread import SUB, source
from os import listdir

from mflow import mflow, PUSH, sleep
from os.path import join, isfile

stop_event = multiprocessing.Event()



def tx_task(bind_address, input_folder, stop_event):
    files = sorted(listdir(input_folder))
    stream = mflow.connect(bind_address, conn_type="bind", mode=PUSH)

    try:
        while not stop_event.is_set():
            for index, raw_file in enumerate(files):
                if not stop_event.is_set():
                    filename = join(input_folder, raw_file)
                    if not (raw_file.endswith('.raw') and isfile(filename)):
                        continue

                    with open(filename, mode='rb') as file_handle:
                        send_more = False
                        if index + 1 < len(files):  # Ensure that we don't run out of bounds
                            send_more = raw_file.split('_')[0] == files[index + 1].split('_')[0]

                        print('Sending %s [%s]' % (raw_file, send_more))
                        stream.send(file_handle.read(), send_more=send_more)
                        time.sleep(0.2)
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