from setuptools import setup
from cam_server_client._version import __VERSION__

PACKAGE_PREFIX = "psi-"
PACKAGE_NAME = "camserver-client"

def version():
    return __VERSION__

setup(
      name=PACKAGE_PREFIX + PACKAGE_NAME,
      version=version(),
      maintainer="Paul Scherrer Institute",
      maintainer_email="daq@psi.ch",
      author="Paul Scherrer Institute",
      author_email="daq@psi.ch",
      description="CamServer client classes.",
      license="GPL3",
      packages=["cam_server_client",],
      zip_safe = False
      )
