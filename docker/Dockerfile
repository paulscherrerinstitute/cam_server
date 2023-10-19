FROM paulscherrerinstitute/cam_server:5.5.5
    RUN conda install -c paulscherrerinstitute -c conda-forge cam_server=5.6.6 && conda clean --all

#FROM paulscherrerinstitute/cam_server:5.4.0
#    RUN conda install -c paulscherrerinstitute -c conda-forge cam_server=5.6.0 && conda clean --all && apt-get update --allow-releaseinfo-change && apt-get -y install make gcc

#OR:
#    FROM paulscherrerinstitute/cam_server:5.2.0
#    RUN conda install -c paulscherrerinstitute -c conda-forge cam_server=5.4.0 lmfit jungfrau_utils=3.12.1


#FROM continuumio/miniconda3:4.6.14
#    RUN conda install -c paulscherrerinstitute -c conda-forge python=3.8 cam_server=5.3.2  lmfit && conda clean --all && apt-get update --allow-releaseinfo-change && apt-get -y install vim


#    RUN conda install -c paulscherrerinstitute -c conda-forge python=3.8 cam_server=5.3.2 && conda clean --all && apt-get update --allow-releaseinfo-change && apt-get -y install vim
#    RUN  conda install -c paulscherrerinstitute -c conda-forge python=3.8 requests bottle psutil paste cheroot numpy scipy numba 'bitshuffle>=0.3.6.1' pyepics  h5py jungfrau_utils 'bsread>1.5.1' pillow matplotlib
#         && conda clean --all && \
#         apt-get update --allow-releaseinfo-change && \
#         apt-get -y install vim


#cam_server=5.3.2 \ matplotlib scikit-image

#conda install -c conda-forge llvmlite=0.38.0=py38h38d86a4_1 numba=0.55.1=py38h4bf6c61_0 tbb=2021.5.0=h924138e_1 && \
#conda install -c conda-forge liblapack=3.9.0=16_linux64_openblas
#conda install -c conda-forge imagecodecs=2022.8.8=py38hf877509_4

#conda update -c paulscherrerinstitute -c conda-forge --all  && \