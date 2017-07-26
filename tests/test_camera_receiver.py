import unittest

from cam_server.camera.receiver import CameraSimulation


class CameraReceiverTest(unittest.TestCase):

    def test_camera_simulation(self):
        camera = CameraSimulation()

        n_images_to_receive = 5

        def callback_method(image, timestamp):
            self.assertIsNotNone(image, "Image should not be None")
            self.assertIsNotNone(timestamp, "Timestamp should not be None")

            nonlocal n_images_to_receive
            if n_images_to_receive <= 0:
                camera.clear_callbacks()
                camera.simulation_stop_event.set()

            n_images_to_receive -= 1

        camera.connect()
        camera.add_callback(callback_method)

        camera.simulation_stop_event.wait()

if __name__ == '__main__':
    unittest.main()
