from setuptools import setup
from cam_server_client import __VERSION__

setup(name="cam_server",
      version=__VERSION__,
      maintainer="Paul Scherrer Institute",
      maintainer_email="daq@psi.ch",
      author="Paul Scherrer Institute",
      author_email="daq@psi.ch",
      description="CamServer client classes.",

      license="GPL3",

      packages=["cam_server_client",
                ],

      zip_safe = False
      )
