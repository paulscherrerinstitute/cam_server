import glob
import logging
import os
import sys
from importlib import import_module

from setuptools import sandbox

_logger = logging.getLogger(__name__)

def load_from_source(file_name):
    mod_name=os.path.splitext(os.path.basename(file_name))[0]
    mod_path = os.path.dirname(file_name)
    mod_ver  = "1.0"
    temp_path = mod_path + "/temp"

    module_src_str="""
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <stdlib.h>

static PyObject *moduleErr;

PyObject *process(PyObject *self, PyObject *args);

static PyMethodDef methods[] = {
    {"process",  process, METH_VARARGS,  "Processing function."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};
static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, \"""" + mod_name + """\", NULL, -1, methods
};

PyMODINIT_FUNC PyInit_""" + mod_name + """(void)
{
    PyObject *m;

    m = PyModule_Create(&module);
    if (m == NULL)
        return NULL;
    import_array();

    char errname[100] = \"""" + mod_name + """\";
    strcat(errname, ".error");
    moduleErr = PyErr_NewException(errname, NULL, NULL);
    Py_XINCREF(moduleErr);
    if (PyModule_AddObject(m, "error", moduleErr) < 0) {
        Py_XDECREF(moduleErr);
        Py_CLEAR(moduleErr);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}    
    """
    module_src_file = mod_path + "/" + "module.c"
    with open(module_src_file, "w") as text_file:
        text_file.write(module_src_str)

    setuppy_str = """
from distutils.core import setup, Extension
import numpy
module1 = Extension('""" + mod_name + """', sources = ['""" + mod_name + """.c'])
setup (name = '""" + mod_name + """',
       version = '""" + mod_ver + """',
       description = 'Processing pipeline """ + mod_name + """ compiled as a C extension',
       include_dirs=[numpy.get_include()],
       ext_modules = [module1])
"""
    setuppy_file = mod_path + "/setup_" + mod_name + ".py"
    with open(setuppy_file, "w") as text_file:
        text_file.write(setuppy_str)

    try:
        sandbox.run_setup(setuppy_file, ['build_ext', '-b', mod_path])
    except Exception as e:
        _logger.exception("Error compiling module: " + mod_name + ": " + str(e))
    finally:
        os.remove(module_src_file)
        os.remove(setuppy_file)
        os.rmdir(temp_path)

    if mod_path not in sys.path:
        sys.path.append(mod_path)
    try:
        mod = import_module(mod_name)
    except Exception as e:
        _logger.exception("Error importing  module: " + mod_name + ": " + str(e))
    return mod

def load_from_library(file_name):
    mod_name=os.path.splitext(os.path.basename(file_name))[0].split(".")[0]
    mod_path = os.path.dirname(file_name)
    if mod_path not in sys.path:
        sys.path.append(mod_path)
    try:
        mod = import_module(mod_name)
    except Exception as e:
        _logger.exception("Error loading module: " + mod_name + ": " + str(e))
    return mod


def get_file_extension(file):
    try:
        ext = os.path.splitext(file)[1][1:]
        if ext: return ext
    except:
        pass
    return None

def load_module(name, path):
    path = os.path.abspath(path)
    ext = get_file_extension(name)
    if ext == "so":
        lib_name = path + "/" + name
        src_name = ""
    elif ext == "c":
        lib_name = ""
        src_name = path + "/" + name
    else:
        file_list = glob.glob(path + "/" + name + "*.so")
        lib_name = "" if len(file_list) == 0 else file_list[0]
        src_name = path + "/" + name + ".c"

    lib_exists = os.path.exists(lib_name)
    src_exists = os.path.exists(src_name)
    lib_time = os.path.getmtime(lib_name) if lib_exists else 0
    src_time = os.path.getmtime(src_name) if src_exists else 0
    newer_source = src_exists and (src_time>lib_time)

    if lib_exists and not newer_source:
        _logger.info("Loading module from : " + lib_name)
        return load_from_library(lib_name)
    elif src_exists:
        if lib_exists and newer_source:
            _logger.info("Module source file changed - compiling from : " + src_name)
            return load_from_source(src_name)
        else:
            _logger.info("Module libray not present - compiling from : " + src_name)
            return load_from_source(src_name)
    else:
        _logger.warning("Invalid module: " + name)
        raise Exception("Invalid module: " + name)





"""
mod_name = "cpip"
cpip = load_module(mod_name, mod_path)
status = cpip.process("ls -l")
print (status)
"""

"""
mod_path = "/Users/gobbo_a/dev/cam_server/module"
from cam_server.pipeline.configuration import PipelineConfig
from tests import get_simulated_camera
import time
simulated_camera = get_simulated_camera(path="../tests/camera_config/")
image = simulated_camera.get_image()
x_axis, y_axis = simulated_camera.get_x_y_axis()

parameters = PipelineConfig("test_pipeline", {
    "camera_name": "simulation"
}).get_configuration()

pid = 23
timestamp = time.time()
mod_name = "pipproc"
pipproc = load_module(mod_name, mod_path)
parameters["int"]=3
print (image[0][0])
print (image[5][10])
#def process_image(image, pulse_id, timestamp, x_axis, y_axis, parameters, bsdata=None):
result = pipproc.process(image, pid, timestamp, x_axis, y_axis, parameters, None)
print(result)

mod_name = "pipstrm"
pipstrm = load_module(mod_name, mod_path)
data={}
data["image"]=image
data["x_axis"]=x_axis
data["y_axis"]=y_axis
#def process(data, pulse_id, timestamp, params):
result = pipstrm.process(data, pid, timestamp, parameters)
print(result)


"""