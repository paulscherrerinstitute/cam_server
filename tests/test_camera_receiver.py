import unittest

from cam_server.camera.receiver import CameraSimulation


class CameraReceiverTest(unittest.TestCase):

    def test_camera_simulation(self):
        camera = CameraSimulation()

        def callback_method(image, timestamp):

            self.assertIsNotNone(image, "Image should not be None")
            self.assertIsNotNone(timestamp, "Timestamp should not be None")

            camera.clear_callbacks()
            camera.disconnect()

        camera.add_callback(callback_method)
        camera.connect()


if __name__ == '__main__':
    unittest.main()
