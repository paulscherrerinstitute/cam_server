import os
import signal
from time import sleep
from cam_server.camera.source.simulation import CameraSimulation
from cam_server.camera.configuration import CameraConfig

def require_folder(path):
    if not os.path.isdir(path):
        os.mkdir(path)


def test_cleanup(servers=[], processes=[], files=[]):
    for s in servers:
        try:
            s.stop_all_instances()
        except:
            pass
    sleep(1.0)

    for p in processes:
        try:
            os.kill(p.pid, signal.SIGINT)
        except:
            pass

    for p in processes:
        try:
            p.join(2.0)
        except:
            pass

    for p in processes:
        if p.exitcode is None:
            print("Cannot stop process: ", p.pid)
        else:
            print("Stopped process: ", p.pid)

    for f in files:
        try:
            if os.path.isfile(f):
                os.remove(f)
            elif os.path.isdir(f):
                os.rmdir(f)
        except:
            pass

    # Wait for the server to die.
    #sleep(sleep_time)

def is_port_available(port):
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind((socket.gethostname(), port))
        return True
    except:
        return False
    finally:
        try:
            s.close()
        except:
            pass

def get_config(path, name):
    from cam_server.instance_management.configuration import ConfigFileStorage
    cfg = ConfigFileStorage(path)
    return cfg.get_config(name)


def get_simulated_camera(path="camera_config/",  name = "simulation"):
    return CameraSimulation(CameraConfig(name,get_config(path, name)))
