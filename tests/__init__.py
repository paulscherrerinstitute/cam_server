import os
import signal
from time import sleep


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
            os.remove(f)
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
