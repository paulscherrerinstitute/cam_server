import time
import unittest

from cam_server.loader import *
from cam_server.pipeline.configuration import PipelineConfig
from tests import get_simulated_camera


class LoaderTest(unittest.TestCase):
    def setUp(self):
        test_base_dir = os.path.split(os.path.abspath(__file__))[0]
        self.mod_path = test_base_dir + "/modules"
        self.simulated_camera = get_simulated_camera(path="../tests/camera_config/")

    def tearDown(self):
        pass

    def test_loader(self):
        image = self.simulated_camera.get_image()
        x_axis, y_axis = self.simulated_camera.get_x_y_axis()

        parameters = PipelineConfig("test_pipeline", {
            "camera_name": "simulation"
        }).get_configuration()

        pid = 23
        timestamp = time.time()
        mod_name = "pipproc"
        pipproc = load_module(mod_name, self.mod_path)
        parameters["int"] = 3
        print(image[0][0])
        print(image[5][10])
        # def process_image(image, pulse_id, timestamp, x_axis, y_axis, parameters, bsdata=None):
        result = pipproc.process(image, pid, timestamp, x_axis, y_axis, parameters, None)
        print(result)

        mod_name = "pipstrm"
        pipstrm = load_module(mod_name, self.mod_path)
        data = {}
        data["image"] = image
        data["x_axis"] = x_axis
        data["y_axis"] = y_axis
        # def process(data, pulse_id, timestamp, params):
        result = pipstrm.process(data, pid, timestamp, parameters)
        print(result)


if __name__ == '__main__':
    unittest.main()
