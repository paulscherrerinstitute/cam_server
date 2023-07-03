#!/bin/bash

# Create bin files

cat > /opt/python/bin/start_pipeline_manager << EOF
#!/bin/bash

/opt/python/bin/python /opt/python/cam_server/cam_server/start_pipeline_manager.py \
	-b /cam_server_configuration/configuration/pipeline_config \
        -g /cam_server_configuration/configuration/background_config \
        -u /cam_server_configuration/configuration/user_scripts \
	--log_level=INFO \
	--web_server=cherrypy \
	--client_timeout=10.0 \
	--update_timeout=2.0 \
        --background_files_days_to_live=356
	${1+"$@"}
EOF


cat > /opt/python/bin/start_camera_manager << EOF
#!/bin/bash

/opt/python/bin/python /opt/python/cam_server/cam_server/start_camera_manager.py \
	-b /cam_server_configuration/configuration/camera_config \
	-u /cam_server_configuration/configuration/user_scripts \
	--log_level=INFO \
	--web_server=cherrypy \
	--client_timeout=10.0 \
	--update_timeout=2.0 \
	${1+"$@"}
EOF


chmod 755  /opt/python/bin/start_pipeline_manager
chmod 755  /opt/python/bin/start_camera_manager


# Create service files

#!/bin/bash

cat > /etc/systemd/system/camera_manager.service << EOF
[Unit]
Description=camera_manager service

[Install]
WantedBy=default.target

[Service]
TimeoutStartSec=0
ExecStart=/opt/python/bin/start_camera_manager
Restart=always
EOF


cat > /etc/systemd/system/pipeline_manager.service << EOF
[Unit]
Description=pipeline_manager service

[Install]
WantedBy=default.target

[Service]
TimeoutStartSec=0
ExecStart=/opt/python/bin/start_pipeline_manager
Restart=always
EOF


# Enable and start services

systemctl daemon-reload
systemctl enable camera_manager pipeline_manager
systemctl restart camera_manager pipeline_manager