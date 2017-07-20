from cam_server.instance_management.wrapper import InstanceWrapper


class CameraInstanceWrapper(InstanceWrapper):
    def __init__(self, process_function, camera, stream_port):

        super(CameraInstanceWrapper, self).__init__(camera.get_name(), process_function,
                                                    camera, stream_port)

        self.camera = camera
        # TODO: Retrieve real address.
        self.stream_address = "tcp://%s:%d" % ("127.0.0.1", self.stream_port)

    def get_info(self):
        return {"stream_address": self.stream_address,
                "is_stream_active": self.is_running(),
                "camera_geometry": self.camera.get_geometry(),
                "camera_name": self.camera.get_name()}
