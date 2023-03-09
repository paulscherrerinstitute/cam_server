import os
import unittest

from threading import Thread

import multiprocessing

from bsread import SUB, source
from cam_server.camera.configuration import CameraConfig
from cam_server.camera.sender import process_bsread_camera
from tests.helpers.factory import MockCameraBsread, replay_dump


class CameraBsreadTransceiverTest(unittest.TestCase):
    def test_bsread_transceiver(self):
        manager = multiprocessing.Manager()
        stop_event = multiprocessing.Event()
        statistics = manager.Namespace()
        parameter_queue = multiprocessing.Queue()
        logs_queue = multiprocessing.Queue()

        expected_width = 659
        expected_height = 494
        expected_shape = [expected_width, expected_height]

        mock_camera = MockCameraBsread(CameraConfig("SLG-LCAM-C102"), expected_width, expected_height,
                                       "tcp://0.0.0.0:9999")

        def transceiver():
            process_bsread_camera(stop_event, statistics, parameter_queue, logs_queue, mock_camera, 12000)

        thread1 = Thread(target=transceiver)
        thread1.start()

        test_base_dir = os.path.split(os.path.abspath(__file__))[0]
        thread2 = Thread(target=replay_dump, args=("tcp://0.0.0.0:9999", os.path.join(test_base_dir,
                                                                                      "test_camera_dump")))
        thread2.start()

        with source(host="0.0.0.0", port=12000, mode=SUB) as stream:
            data1 = stream.receive()
            data2 = stream.receive()

        self.assertIsNotNone(data1)
        self.assertIsNotNone(data2)

        stop_event.set()
        thread1.join()
        thread2.join()

        self.assertListEqual(list(data1.data.data["image"].value.shape), expected_shape[::-1])
        self.assertListEqual(list(data2.data.data["image"].value.shape), expected_shape[::-1])

        self.assertEqual(data1.data.data["width"].value, data2.data.data["width"].value)
        self.assertEqual(data1.data.data["height"].value, data2.data.data["height"].value)
        self.assertEqual(data1.data.data["width"].value, expected_width)
        self.assertEqual(data1.data.data["height"].value, expected_height)

        self.assertEqual(data1.data.pulse_id, data2.data.pulse_id - 1)


if __name__ == '__main__':
    unittest.main()
