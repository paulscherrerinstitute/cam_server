from setuptools import setup
import pathlib


def version():
    p = pathlib.Path(__file__).parent.joinpath("cam_server_client").joinpath("package_version.txt")
    with open(p, "r") as f1:
        return f1.read()

setup(name="cam_server",
      version=version(),
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
