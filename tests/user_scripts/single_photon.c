#include "module.c"
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <syslog.h>


const char *CHANNEL_NAMES[] = {"EVENT_NUM", "EVENT_I", "EVENT_J", "EVENT_CHARGE", "EVENT_ETA_X", "EVENT_ETA_Y", "EVENT_I_INTERP", "EVENT_J_INTERP"};


// max number of events per frame
#define MAX_NUM_EVENTS 100
#define EVENT_CHANNELS 7

double evt_p[EVENT_CHANNELS][MAX_NUM_EVENTS];


int func_ph_1d_double( double *frame, int i_dim, int j_dim,  double *th_m);

void setArrayToValue(double array[], int size, double value) {
    for (int i = 0; i < size; i++) {
        array[i] = value;
    }
}

//Initialization: Threshold & background
initialized = 0;
int threshold_num_elements=-1;
double *threshold= NULL;
double *background= NULL;
const int MAX_LINE_LENGTH = 50000;


int parseTextFile(const char * fileName, double *arr, int size_x, int size_y){
        FILE *file = fopen(fileName, "rb");
        if (file == NULL) {
            syslog(LOG_ERR, "Failed to open data file: %s" , fileName);
            return -1;
        }
        int ret = 1;
        char line[MAX_LINE_LENGTH];
        int x = 0;
        int y = 0;
        while (fgets(line, sizeof(line), file) != NULL) {
            char* token = strtok(line, " ");
            while (token != NULL) {
                threshold[y * size_x + x] = atof(token);
                token = strtok(NULL, " ");
                x++;
            }
            y++;
            if (x != size_x){
                syslog(LOG_ERR, "Invalid data file - wrong number of columns: %s" , fileName);
                ret = -2;
                break;
            }
            x = 0;
        }
        if (y != size_y){
            syslog(LOG_ERR, "Invalid data file - wrong number of rows: %s" , fileName);
            ret = -3;
        }
        fclose(file);
        return ret;
}

double getParDouble(PyObject *pars, char *name, double defaultValue){
    PyObject* threshold_obj = PyDict_GetItemString(pars, name);
    if (threshold_obj!=NULL){
         if (PyFloat_Check(threshold_obj)) {
            return PyFloat_AsDouble(threshold_obj);
        } else if (PyLong_Check(threshold_obj)) {
            return PyLong_AsDouble(threshold_obj);
        }
    }
    return defaultValue;
}

int initialize(int size_x, int size_y, PyObject *pars){
    openlog("single_photon", LOG_PID, LOG_USER);
    syslog(LOG_INFO, "Initialized single_photon.");
    int ret = 1;

    threshold = (double *)malloc(size_x*size_y*sizeof(double));
    double threshold_val = getParDouble(pars, "threshold", 60000.0);
    PyObject* threshold_file = PyDict_GetItemString(pars, "threshold_file");
    if (threshold_file!=NULL){
        const char * threshold_file_str = PyUnicode_AsUTF8(threshold_file);
        int ret = parseTextFile(threshold_file_str, threshold, size_x, size_y);
        if (ret<0){
           setArrayToValue(threshold,size_x*size_y, threshold_val);
        }
    } else {
       setArrayToValue(threshold,size_x*size_y, threshold_val);
    }


    //background (all matrices are indexed in 1d)
    background = malloc(size_x*size_y*sizeof(double));
    double background_val =  getParDouble(pars, "background", 0.0);
    PyObject* background_file = PyDict_GetItemString(pars, "background_file");
    if (background_file!=NULL){
        const char * background_file_str = PyUnicode_AsUTF8(background_file);
        int ret = parseTextFile(background_file_str, background, size_x, size_y);
        if (ret<0){
           setArrayToValue(background,size_x*size_y, background_val);
        }
    } else {
       setArrayToValue(background,size_x*size_y, background_val);
    }
    return ret;
}




//def process_image(image, pulse_id, timestamp, x_axis, y_axis, parameters, bsdata=None):
PyObject *process(PyObject *self, PyObject *args)
{
    PyArrayObject *image;
    long pulse_id;
    PyObject /*double*/ *timestamp;
    long seconds, nanos;
    PyArrayObject *x_axis;
    PyArrayObject *y_axis;
    PyObject *pars;
    PyObject *bsdata;

    //if (!PyArg_ParseTuple(args, "OldOOO|O", &image, &pulse_id, &timestamp, &x_axis, &y_axis, &pars, &bsdata))
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


    //Initialization
    if (initialized==0){
        initialized = initialize(size_x, size_y, pars);
    }
    int i,j,l;
    int i_dim=size_y;
    int j_dim=size_x;

    double *frameBKsub = malloc(i_dim*j_dim*sizeof(double));
    for(i=0; i<i_dim;i++) {
      for(j=0;j<j_dim;j++) {
              frameBKsub[i*j_dim+j]=(double)img_data[i*j_dim+j] - background[i*j_dim+j];
          }
       }
    int evnt_num = func_ph_1d_double( frameBKsub, i_dim, j_dim, threshold);
//    int evnt_num = func_ph_1d_double( (double)img_data, i_dim, j_dim, threshold);

    //Create return dictionary
    PyObject *ret = PyDict_New();
    PyObject* camera_name = PyDict_GetItemString(pars, "camera_name");
    const char * camera_name_str = PyUnicode_AsUTF8(camera_name);
    char channel_name[200];
    //PyDict_SetItemString(ret, "camera_name", camera_name);
    sprintf(channel_name, "%s:%s",camera_name_str,  CHANNEL_NAMES[0]);
    PyDict_SetItemString(ret, channel_name, PyLong_FromLong(evnt_num));

//PyDict_SetItemString(ret,"T0", PyFloat_FromDouble(threshold[0]));
//PyDict_SetItemString(ret, "Init", PyLong_FromLong(initialized));
    for (int i=0; i<EVENT_CHANNELS; i++){
        sprintf(channel_name, "%s:%s",camera_name_str,  CHANNEL_NAMES[i+1]);
        npy_intp arr_dims[1] = {MAX_NUM_EVENTS};
        //setArrayToValue(evt_p[i],MAX_NUM_EVENTS, i+10.0);
        PyObject *parr =PyArray_SimpleNewFromData(1, arr_dims, NPY_DOUBLE, evt_p[i]);
        PyDict_SetItemString(ret, channel_name, parr);
     }
     //for (l = 0; l < EVENT_CHANNELS; l++) {
     //  free(evns1d.evnt_ijc[l]);
     //}
     //free(evns1d.evnt_ijc);
     free(frameBKsub);
    return ret;
}

 int func_ph_1d_double( double *frame, int i_dim, int j_dim,  double *th_m)
 {
   //int th= 50;
   int i, j, l=0, m=0, n=0, evt_m_i=0, evt_i=0 ;
   int dist_i, dist_j;
   double charge_evt;
   int th;

   //double **evt_p= (double**)malloc(EVENT_CHANNELS * sizeof(double*));
   //for (l = 0; l < EVENT_CHANNELS; l++) {
   //    evt_p[l] = (double*)malloc(MAX_NUM_EVENTS * sizeof(double));}

   /*Counter variables for the loop*/
   double charge = 0;
   double eta_x =  0;
   double eta_y =  0;
   double i_interp = 0;
   double j_interp = 0;
   double C[]= { 4.37805097e-03, 2.43266401e-01, -7.81479328e+00, 7.70533057e+01, -2.37906124e+02, 3.42988113e+02, -2.35979731e+02, 6.24074562e+01};
   double D[]= {-2.05180773e-04, 4.35314696e-01, -1.32557223e+01, 1.19990373e+02, -3.80696758e+02, 5.81427096e+02, -4.30789104e+02, 1.23894761e+02};

   for(i=0; i<i_dim-1; i++) {
          if (evt_i>=MAX_NUM_EVENTS){
               break;
          }
      for(j=0;j<j_dim-1;j++) {
          if (evt_i>=MAX_NUM_EVENTS){
               break;
          }


          // 2x2 version
          charge = frame[i*j_dim+j]+frame[(i+1)*j_dim+j] + frame[i*j_dim+(j+1)]+frame[(i+1)*j_dim+j+1];

          //pixel by pixel threshold
          th = th_m[i*j_dim +j];

          //check if charge above threshold
          if(charge>th) {
                 eta_x    = (frame[(i+1)*j_dim + j    ]+frame[(i+1)*j_dim +  (j+1)])/charge;
                 eta_y    = (frame[    i*j_dim + (j+1)]+frame[(i+1)*j_dim +  (j+1)])/charge;
                 i_interp =  i + (C[0] + C[1]*eta_x+ C[2]*pow(eta_x,2) + C[3]*pow(eta_x,3) + C[4]*pow(eta_x,4)+ C[5]*pow(eta_x,5) + C[6]*pow(eta_x,6) + C[7]*pow(eta_x,7));
                 j_interp =  j + (D[0] + D[1]*eta_y+ D[2]*pow(eta_y,2) + D[3]*pow(eta_y,3) + D[4]*pow(eta_y,4)+ D[5]*pow(eta_y,5) + D[6]*pow(eta_y,6) + D[7]*pow(eta_y,7));

                 // 1st case: first event
                 if(evt_i==0){
                 evt_p[0][evt_i] = i;
                 evt_p[1][evt_i] = j;
                 evt_p[2][evt_i] = charge;
                 evt_p[3][evt_i] = eta_x;
                 evt_p[4][evt_i] = eta_y;
                 evt_p[5][evt_i] = i_interp; //
                 evt_p[6][evt_i] = j_interp; //
                 evt_i++;

                 } else {
                 // 2nd case: not 1st event. we check if it is a neighbourg of the previos events and if charge is larger
                 n=0;
                 evt_m_i = evt_i;
                 for(m=0; m<evt_m_i; m++) {
                        dist_i     = abs(evt_p[0][evt_i-1-m] - i); //fix here
                        dist_j     = abs(evt_p[1][evt_i-1-m] - j);
                        charge_evt =     evt_p[2][evt_i-1-m];

                        if( dist_i< 2 && dist_j<2) {
                            if(charge_evt < charge) {
                             //2nd case: not 1st event, but neigboor of previos event, this event has more charge
                             evt_p[0][evt_i-1-m] = i;
                             evt_p[1][evt_i-1-m] = j;
                             evt_p[2][evt_i-1-m] = charge;
                             evt_p[3][evt_i-1-m] = eta_x;
                             evt_p[4][evt_i-1-m] = eta_y;
                             evt_p[5][evt_i-1-m] = i_interp;
                             evt_p[6][evt_i-1-m] = j_interp;
                            } else {
                                //3d case not 1st event, but neigboor of previos event, this event has less charge
                            }

                        } else {
                             // not a neighbor of the m event

                             n++;

                             //now we check if it is not a neighborg of any previous event
                             if((n)==evt_i) {

                             evt_p[0][evt_i] = i;
                             evt_p[1][evt_i] = j;
                             evt_p[2][evt_i] = charge;
                             evt_p[3][evt_i] = eta_x;
                             evt_p[4][evt_i] = eta_y;
                             evt_p[5][evt_i] = i_interp; //
                             evt_p[6][evt_i] = j_interp; //
                             evt_i++;
                             }
                        }
                   }
            }
        }
      }
    }
    return evt_i;
 }

