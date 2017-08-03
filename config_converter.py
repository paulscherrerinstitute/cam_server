import argparse
import glob
import json
import os
from collections import OrderedDict
from os.path import basename


def convert_config(old_base_dir, new_cam_base_dir, new_pipeline_base_dir):
    for old_config_file in glob.glob(old_base_dir + '/*.json'):
        with open(old_config_file) as data_file:
            old_config = json.load(data_file)

        prefix = old_config["camera"]["prefix"]
        mirror_x = old_config["camera"].get("mirror_x", False)
        mirror_y = old_config["camera"].get("mirror_y", False)
        rotate = old_config["camera"].get("rotate", 0)

        config_name = os.path.splitext(basename(old_config_file))[0]

        new_cam_config = OrderedDict({"name": config_name,
                                      "prefix": prefix,
                                      "mirror_x": mirror_x,
                                      "mirror_y": mirror_y,
                                      "rotate": rotate})

        cam_config_filename = os.path.join(new_cam_base_dir, config_name + ".json")
        with open(cam_config_filename, 'w') as outfile:
            json.dump(new_cam_config, outfile, indent=4)

        new_pipeline_config = OrderedDict({"name": config_name,
                                           "camera_name": config_name})

        calibration = old_config["camera"].get("calibration")
        new_calibration = {}
        if calibration:
            new_calibration["reference_marker"] = calibration.get("reference_marker", [0, 0, 100, 100])
            new_calibration["reference_marker_width"] = calibration.get("reference_marker_width", 100.0)
            new_calibration["reference_marker_height"] = calibration.get("reference_marker_height",100.0)
            new_calibration["angle_horizontal"] = calibration.get("horizontal_camera_angle", 0.0)
            new_calibration["angle_vertical"] = calibration.get("vertical_camera_angle", 0.0)

            new_pipeline_config["camera_calibration"] = new_calibration

        pipeline_config_filename = os.path.join(new_pipeline_base_dir, config_name + ".json")
        with open(pipeline_config_filename, 'w') as outfile:
            json.dump(new_pipeline_config, outfile, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert old to new config format.')
    parser.add_argument('--old_base_dir', default="configuration", help="Old config base directory")
    parser.add_argument('--new_cam_base_dir', default="configuration/camera", help="New camera base directory.")
    parser.add_argument('--new_pipeline_base_dir', default="configuration/pipeline",
                        help="New pipeline base directory.")

    arguments = parser.parse_args()

    convert_config(arguments.old_base_dir, arguments.new_cam_base_dir, arguments.new_pipeline_base_dir)
