FROM osrf/ros:noetic-desktop-full

RUN apt-get update && apt-get install -y \
        ros-noetic-urdf \
        ros-noetic-kdl-parser \
        ros-noetic-urdf-parser-plugin \
        ros-noetic-hardware-interface \
        ros-noetic-controller-manager \
        ros-noetic-controller-interface \
        ros-noetic-controller-manager-msgs \
        ros-noetic-control-msgs \
        ros-noetic-ros-control \
        "ros-noetic-gazebo-*" \
        "ros-noetic-robot-state-*" \
        "ros-noetic-joint-state-*" \
        ros-noetic-rqt-gui \
        ros-noetic-rqt-controller-manager \
        "ros-noetic-plotjuggler*" \
        cmake \
        build-essential \
        libpcl-dev \
        libeigen3-dev \
        libopencv-dev \
        libmatio-dev \
        python3-pip \
        libboost-all-dev \
        libtbb-dev \
        liburdfdom-dev \
        liborocos-kdl-dev \
        libspdlog-dev \
        git \
        tree \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/include/eigen3/Eigen /usr/include/Eigen || true

WORKDIR /root/tron1_ws

ENV ROBOT_TYPE=SF_TRON1A

RUN echo 'source /opt/ros/noetic/setup.bash' >> /root/.bashrc && \
    echo 'export ROBOT_TYPE=SF_TRON1A' >> /root/.bashrc

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
