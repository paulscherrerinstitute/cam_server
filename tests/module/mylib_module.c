#define LIBRARY_NAME "mylib"
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <errno.h>


PyObject *process(PyObject *self, PyObject *args);

static PyObject *moduleErr;

static PyMethodDef methods[] = {
    {"process",  process, METH_VARARGS,  "Processing function."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};
static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, LIBRARY_NAME, NULL, -1, methods
};

PyMODINIT_FUNC PyInit_mylib(void)
{
    PyObject *m;
    m = PyModule_Create(&module);
    if (m == NULL)
        return NULL;
    import_array();

    char errName[100]; sprintf(errName, "%s.error", LIBRARY_NAME);
    moduleErr = PyErr_NewException(errName, NULL, NULL);
    Py_XINCREF(moduleErr);
    if (PyModule_AddObject(m, "error", moduleErr) < 0) {
        Py_XDECREF(moduleErr);
        Py_CLEAR(moduleErr);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}
