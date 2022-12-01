#include "module.c"
#include <math.h>

//def process(data, pulse_id, timestamp, params):
PyObject *process(PyObject *self, PyObject *args)
{
    long pulse_id;
    double timestamp;
    PyObject *pars;
    PyObject *data;

    if (!PyArg_ParseTuple(args, "OldO", &data, &pulse_id, &timestamp, &pars))
        return NULL;
    if (pulse_id < 0) {
        PyErr_SetString(moduleErr, "Invalid Pulse ID");
        return NULL;
    }

    PyObject* camera_name = PyDict_GetItemString(pars, "camera_name");
    PyArrayObject* image = (PyArrayObject *)PyDict_GetItemString(data, "image");

    //Acessing image
    printf("Element size %d\n",  image->descr->elsize);
    printf("Dims %d\n",  image->nd);
    int size_x=image->dimensions[1]; printf("Size X %d\n",  size_x);
    int size_y=image->dimensions[0]; printf("Size Y %d\n",  size_y);
    unsigned short* sdata = (unsigned short*)image->data;

    int min_val=0x10000;
    int max_val=0;
    int sum=0;
    for (int y=0; y<size_y;y++){
        for (int x=0; x<size_x; x++){
            unsigned short val = sdata[y*size_x + x];
            sum+=val;
            max_val = fmax(max_val, val);
            min_val = fmin(min_val, val);
        }
    }


    PyObject *ret = PyDict_New();
    PyDict_SetItemString(ret, "camera_name", camera_name);
    PyDict_SetItemString(ret, "max", PyLong_FromLong(max_val));
    PyDict_SetItemString(ret, "min", PyLong_FromLong(min_val));
    PyDict_SetItemString(ret, "sum", PyLong_FromLong(sum));
    return ret;

    //return camera_name;

}