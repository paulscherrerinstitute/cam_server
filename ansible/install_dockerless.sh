#!/bin/bash

export PY_MAJOR=3
export PY_MINOR=8
export PY_PATCH=17

export AFFINITY=2-23,48-71

export PY_VER="${PY_MAJOR}.${PY_MINOR}.${PY_PATCH}"
export PY_MIN="${PY_MAJOR}.${PY_MINOR}"


#Default folders
mkdir -p /cam_server_configuration
mkdir -p /cam_server_configuration/configuration
mkdir -p /cam_server_configuration/configuration/tmp
mkdir -p /cam_server_configuration/configuration/tmp/user_scripts
mkdir -p /cam_server_configuration/configuration/tmp/background_config
mkdir -p /tmp/cam_server


#Python Dependencies
yum install -y gcc openssl-devel bzip2-devel libffi-devel zlib-devel
mkdir -p /opt/python
mkdir -p /opt/python/bin


#Install Python
cd /opt/python
wget -O Python-${PY_VER}.tgz https://www.python.org/ftp/python/${PY_VER}/Python-${PY_VER}.tgz
tar xzf Python-${PY_VER}.tgz
rm Python-${PY_VER}.tgz
cd /opt/python/Python-${PY_VER}
./configure --enable-optimizations
make altinstall

#Install Python packages
/usr/local/bin/python${PY_MIN} -m pip install --upgrade pip
/usr/local/bin/pip${PY_MIN} install numpy scipy numba requests bottle psutil paste cheroot pyepics h5py pillow matplotlib scikit-image lmfit click pyzmq redis
/usr/local/bin/python${PY_MIN} -m pip install urllib3==1.26.6


#Clone PSI repos
cd /opt/python
git clone https://github.com/paulscherrerinstitute/cam_server.git
git clone https://github.com/paulscherrerinstitute/bsread_python.git
git clone https://github.com/paulscherrerinstitute/mflow.git
git clone https://github.com/paulscherrerinstitute/jungfrau_utils.git


# Install PSI bitshuffle - avoid the multithreading bug consuming all cores
wget -O bitshuffle.tar.bz2 https://anaconda.org/paulscherrerinstitute/bitshuffle/0.3.6.2/download/linux-64/bitshuffle-0.3.6.2-py${PY_MAJOR}${PY_MINOR}h6bb024c_0.tar.bz2
tar xvf ./bitshuffle.tar.bz2
cp -r ./lib/python${PY_MIN}/site-packages/bitshuffle /usr/local/lib/python${PY_MIN}/site-packages/
cp -r ./lib/python${PY_MIN}/site-packages/bitshuffle-0.3.6.dev1-py${PY_MIN}.egg-info /usr/local/lib/python${PY_MIN}/site-packages/
rm bitshuffle.tar.bz2
rm -r ./lib
rm -r ./info


# Create bin files
cat > /opt/python/bin/python << EOF
#!/bin/bash

export PYTHONPATH="\${PYTHONPATH}:/opt/python/mflow:/opt/python/bsread_python:/opt/python/cam_server/:/opt/python/jungfrau_utils"
export EPICS_CA_ADDR_LIST="172.26.0.255 172.26.2.255 172.26.8.255 172.26.16.255 172.26.24.255 172.26.32.255 172.26.40.255 172.26.110.255 172.26.111.255 172.26.120.255 172.27.0.255 saresa-cagw.psi.ch:5062 saresb-cagw.psi.ch:5062 saresc-cagw.psi.ch:5062 satese-cagw.psi.ch:5062 satesf-cagw.psi.ch:5062"
export EPICS_CA_AUTO_ADDR_LIST=NO
export EPICS_CA_MAX_ARRAY_BYTES=40000000

/usr/local/bin/python${PY_MIN}  \${1+"\$@"}
EOF

cat > /opt/python/bin/start_pipeline_worker << EOF
#!/bin/bash

/opt/python/bin/python /opt/python/cam_server/cam_server/start_pipeline_worker.py \
        -g /cam_server_configuration/configuration/tmp/background_config \
        -u /cam_server_configuration/configuration/tmp/user_scripts \
        -c http://sf-daqsync-01.psi.ch:8888 \
        --log_level=INFO \
        --web_server=cherrypy \
        \${1+"\$@"}
EOF


cat > /opt/python/bin/start_camera_worker << EOF
#!/bin/bash

/opt/python/bin/python /opt/python/cam_server/cam_server/start_camera_worker.py \
        -u /cam_server_configuration/configuration/tmp/user_scripts \
        --log_level=INFO \
        --web_server=cherrypy \
        --ipc_feed_folder=/tmp/cam_server \
        \${1+"\$@"}
EOF


chmod 755  /opt/python/bin/python
chmod 755  /opt/python/bin/start_pipeline_worker
chmod 755  /opt/python/bin/start_camera_worker


# Create service files

#!/bin/bash

cat > /etc/systemd/system/camera_worker.service << EOF
[Unit]
Description=camera_worker service

[Install]
WantedBy=default.target

[Service]
TimeoutStartSec=0
CPUAffinity=${AFFINITY}
ExecStart=/opt/python/bin/start_camera_worker
Restart=always
EOF


cat > /etc/systemd/system/pipeline_worker.service << EOF
[Unit]
Description=pipeline_worker service

[Install]
WantedBy=default.target

[Service]
TimeoutStartSec=0
CPUAffinity=${AFFINITY}
ExecStart=/opt/python/bin/start_pipeline_worker
Restart=always
EOF


# Enable and start services

systemctl daemon-reload
systemctl enable camera_worker pipeline_worker
systemctl restart camera_worker pipeline_worker