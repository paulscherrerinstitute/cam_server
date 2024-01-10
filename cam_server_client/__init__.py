from pkg_resources import resource_stream

# Import the cam client.
from cam_server_client.camera_client import CamClient
from cam_server_client.pipeline_client import PipelineClient
from cam_server_client.proxy_client import ProxyClient



def version():
        with resource_stream(__name__, "package_version.txt") as res:
            return res.read().decode()

__VERSION__ = version()
