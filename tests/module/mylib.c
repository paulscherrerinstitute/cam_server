#include "mylib_module.c"
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <stdlib.h>

extern PyObject *moduleErr;
//def process_image(image, pulse_id, timestamp, x_axis, y_axis, parameters, bsdata=None):
PyObject *process(PyObject *self, PyObject *args)
{
    //Parse arguments
    PyArrayObject *image;
    long pulse_id;
    PyObject /*double*/ *timestamp;
    long seconds, nanos;
    PyArrayObject *x_axis;
    PyArrayObject *y_axis;
    PyObject *pars;
    PyObject *bsdata;

    if ( !PyArg_ParseTuple(args, "OlOOOO|O", &image, &pulse_id, &timestamp, &x_axis, &y_axis, &pars, &bsdata) ||
         !PyArg_ParseTuple(timestamp, "ll", &seconds, &nanos) )
        return NULL;

    if (pulse_id < 0) {
        PyErr_SetString(moduleErr, "Invalid Pulse ID");
        return NULL;
    }

    //Acessing image
    int element_size = image->descr->elsize;
    int dims = image->nd;
    int size_x = image->dimensions[1];
    int size_y = image->dimensions[0];
    unsigned short* img_data = (unsigned short*)image->data;


    //generating profile X and intensity
    long intensity=0;
    npy_intp dims_profile[1] = {size_x};
    PyArrayObject *arr_profile =  (PyArrayObject *) PyArray_SimpleNew(1, dims_profile, NPY_FLOAT);
    float* pprofile = (float*)arr_profile->data;
    for (int x=0; x<size_x; x++){
        pprofile[x]=0;
        for (int y=0; y<size_y;y++){
             pprofile[x] += img_data[y*size_x + x];
        }
        intensity += pprofile[x];
    }

    //Acessing parameter dict and
    PyObject* camera_name = PyDict_GetItemString(pars, "camera_name");
    const char * camera_name_str = PyUnicode_AsUTF8(camera_name);

    //create channel names
    char channel_name_profile[100];
    sprintf(channel_name_profile, "%s:%s", camera_name_str, "profile_x");
    char channel_name_intensity[100];
    sprintf(channel_name_intensity, "%s:%s", camera_name_str, "intensity");

    //Create return dictionary
    PyObject *ret = PyDict_New();
    PyDict_SetItemString(ret, channel_name_profile, (PyObject *)arr_profile);
    PyDict_SetItemString(ret, channel_name_intensity, PyLong_FromLong(intensity));
    return ret;
}
