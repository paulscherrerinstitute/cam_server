import argparse

import epics
from time import time, sleep


def test_epics_source(epics_pv, acquisition_time,
                      acquisition_initial_wait_time=1, acquisition_update_interval=0.2, epics_connection_timeout=1):

    print("Testing epics PV %s for %d seconds." % (epics_pv, acquisition_time))

    channel_image = epics.PV(epics_pv, auto_monitor=True)
    channel_image.wait_for_connection(epics_connection_timeout)

    if not channel_image.connected:
        raise RuntimeError("Could not connect to PV %s in time (%d seconds)." % (epics_pv, epics_connection_timeout))

    messages = []

    def process_message(value, timestamp, status, **kwargs):
        messages.append(time())

    channel_image.add_callback(process_message)

    start_time = time()
    current_time = time()

    sleep(acquisition_initial_wait_time)

    while (current_time - start_time) < acquisition_time:
        sleep(acquisition_update_interval)

        current_time = time()
        average_fps = len(messages) / (current_time - start_time)

        print("%s update FPS (average): %s" % (epics_pv, average_fps))


def main():
    parser = argparse.ArgumentParser(description='EPICS source refresh rate tester.')
    parser.add_argument("epics_pv", help="EPICS PV to test the monitor on.")
    parser.add_argument("--time", default=10, type=int, help="How many seconds to run the test.")

    arguments = parser.parse_args()

    test_epics_source(arguments.epics_pv, arguments.time)


if __name__ == "__main__":
    main()
