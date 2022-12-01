#include "module.c"
#include <unistd.h>

//def process_image(image, pulse_id, timestamp, x_axis, y_axis, parameters, bsdata=None):
PyObject *process(PyObject *self, PyObject *args)
{
    PyArrayObject *image;
    long pulse_id;
    double timestamp;
    PyArrayObject *x_axis;
    PyArrayObject *y_axis;
    PyObject *pars;
    PyObject *bsdata;

    if (!PyArg_ParseTuple(args, "OldOOO|O", &image, &pulse_id, &timestamp, &x_axis, &y_axis, &pars, &bsdata))
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

    //Generating binned image
    int bin_x = size_x/2;
    int bin_y = size_y/2;
    npy_intp dims_binning[2] = {bin_y, bin_x};
    PyArrayObject *arr_binning= (PyArrayObject *) PyArray_SimpleNew(2, dims_binning, NPY_UINT);
    unsigned int *pbinning = (unsigned int *)arr_binning->data;
    for (int x=0; x<bin_x; x++){
        for (int y=0; y<bin_y;y++){
            long sum = img_data[(y*2)*bin_x +(x*2)] +
                       img_data[(y*2+1)*bin_x +(x*2)] +
                       img_data[(y*2)*bin_x +(x*2+1)] +
                       img_data[(y*2+1)*bin_x +(x*2+1)];
            pbinning[y*bin_x + x] = sum;
        }
    }

    //generating profile X
    npy_intp dims_profile[1] = {size_x};
    PyArrayObject *arr_profile =  (PyArrayObject *) PyArray_SimpleNew(1, dims_profile, NPY_FLOAT);
    float* pprofile = (float*)arr_profile->data;
    for (int x=0; x<size_x; x++){
        pprofile[x]=0;
        for (int y=0; y<size_y;y++){
            pprofile[x] += img_data[y*size_x + x];
        }
    }

    //Acessing parameter dict
    PyObject* camera_name = PyDict_GetItemString(pars, "camera_name");
    const char * camera_name_str = PyUnicode_AsUTF8(camera_name);

    PyObject *ret = PyDict_New();
    PyDict_SetItemString(ret, "camera_name", camera_name);
    PyDict_SetItemString(ret, "camera_name_str", PyUnicode_FromString(camera_name_str));
    PyDict_SetItemString(ret, "first_image_val", PyLong_FromLong(img_data[0]));
    PyDict_SetItemString(ret, "other_image_val", PyLong_FromLong(img_data[5*size_x + 10]));
    PyDict_SetItemString(ret, "first_x_val", PyFloat_FromDouble(((float*)x_axis->data)[0]));
    PyDict_SetItemString(ret, "profile", (PyObject *)arr_profile);
    PyDict_SetItemString(ret, "binning", (PyObject *) arr_binning);
    return ret;
}



/*

typedef struct PyArrayObject {
    PyObject_HEAD
    char *data;
    int nd;
    npy_intp *dimensions;
    npy_intp *strides;
    PyObject *base;
    PyArray_Descr *descr;
    int flags;
    PyObject *weakreflist;
} PyArrayObject;

typedef struct {
    PyObject_HEAD
    PyTypeObject *typeobj;
    char kind;
    char type;
    char byteorder;
    char flags;
    int type_num;
    int elsize;
    int alignment;
    PyArray_ArrayDescr *subarray;
    PyObject *fields;
    PyObject *names;
    PyArray_ArrFuncs *f;
    PyObject *metadata;
    NpyAuxData *c_metadata;
    npy_hash_t hash;
} PyArray_Descr;
*/