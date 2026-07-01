ARG BASE_IMAGE_PLATFORM=linux/amd64
FROM --platform=${BASE_IMAGE_PLATFORM} mirror.gcr.io/osrf/ros:jazzy-desktop

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ARG ROS_DISTRO=jazzy
ENV ROS_DISTRO=${ROS_DISTRO}
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV GZ_VERSION=harmonic
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Replace repositories in both legacy *.list and Ubuntu 24.04/ROS Jazzy
# deb822 *.sources files. The base image currently stores the ROS repository in
# ros2.sources, so editing only *.list silently leaves packages.ros.org active.
RUN set -eux; \
    find -L /etc/apt -type f \( -name '*.list' -o -name '*.sources' \) \
        -exec sed -i --follow-symlinks \
        -e 's|http://packages.ros.org/ros2/ubuntu|https://mirrors.tuna.tsinghua.edu.cn/ros2/ubuntu|g' \
        -e 's|https://packages.ros.org/ros2/ubuntu|https://mirrors.tuna.tsinghua.edu.cn/ros2/ubuntu|g' \
        -e 's|http://archive.ubuntu.com/ubuntu|https://mirrors.tuna.tsinghua.edu.cn/ubuntu|g' \
        -e 's|https://archive.ubuntu.com/ubuntu|https://mirrors.tuna.tsinghua.edu.cn/ubuntu|g' \
        -e 's|http://security.ubuntu.com/ubuntu|https://mirrors.tuna.tsinghua.edu.cn/ubuntu|g' \
        -e 's|https://security.ubuntu.com/ubuntu|https://mirrors.tuna.tsinghua.edu.cn/ubuntu|g' \
        -e 's/^Types: deb deb-src$/Types: deb/' \
        {} +; \
    printf '%s\n' \
        'Acquire::Retries "5";' \
        'Acquire::http::Timeout "60";' \
        'Acquire::https::Timeout "60";' \
        > /etc/apt/apt.conf.d/80-network-retries; \
	    if grep -Rqs 'packages.ros.org/ros2/ubuntu' /etc/apt/sources.list /etc/apt/sources.list.d 2>/dev/null; then \
	        echo 'ERROR: ROS 2 repository was not replaced' >&2; \
	        exit 1; \
	    fi

# Gazebo Harmonic packages and Python bindings live in the OSRF apt repository.
RUN set -eux; \
    apt-get update; \
    apt-get install -y ca-certificates curl gnupg lsb-release; \
    curl -fsSL https://packages.osrfoundation.org/gazebo.gpg \
        -o /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg; \
    codename="$(. /etc/os-release && echo "${VERSION_CODENAME}")"; \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable ${codename} main" \
        > /etc/apt/sources.list.d/gazebo-stable.list; \
    rm -rf /var/lib/apt/lists/*

# Install PX4 SITL, Gazebo, ROS 2, OpenVINS, VNC, and RL dependencies.
RUN set -eux; \
    apt-get update; \
    apt-get install -y \
    astyle \
    bc \
    build-essential \
    ccache \
    cmake \
    cppcheck \
    dmidecode \
    file \
    g++ \
    gcc \
    gdb \
    git \
    gperf \
    gz-harmonic \
    lcov \
    libasio-dev \
    libatlas-base-dev \
    libboost-all-dev \
    libceres-dev \
    libeigen3-dev \
    libexpat-dev \
    libgflags-dev \
    libgoogle-glog-dev \
    libgstreamer-plugins-base1.0-dev \
    libimage-exiftool-perl \
    libopencv-contrib-dev \
    libopencv-dev \
    libprotobuf-dev \
    libssl-dev \
    libsuitesparse-dev \
    libtinyxml2-dev \
    libunwind-dev \
    libxml2-dev \
    libxml2-utils \
    make \
    ninja-build \
    pkg-config \
    protobuf-compiler \
    python3-colcon-common-extensions \
    python3-dev \
    python3-gz-msgs10 \
    python3-gz-transport13 \
    python3-jinja2 \
    python3-jsonschema \
    python3-numpy \
    python3-pip \
    python3-pytest \
    python3-rosdep \
    python3-setuptools \
    python3-tk \
    python3-venv \
    python3-wheel \
    python3-yaml \
    rsync \
    shellcheck \
    sudo \
    tar \
    unzip \
    wget \
    curl \
    dbus-x11 \
    net-tools \
    novnc \
    tigervnc-standalone-server \
    websockify \
    xfce4 \
    xfce4-goodies \
    xfce4-terminal \
    zip \
    cppzmq-dev \
    gstreamer1.0-libav \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-ugly \
    ros-${ROS_DISTRO}-cv-bridge \
    ros-${ROS_DISTRO}-geometry-msgs \
    ros-${ROS_DISTRO}-image-transport \
    ros-${ROS_DISTRO}-message-filters \
    ros-${ROS_DISTRO}-nav-msgs \
    ros-${ROS_DISTRO}-rclcpp \
    ros-${ROS_DISTRO}-ros-gz-bridge \
    ros-${ROS_DISTRO}-ros-gz-interfaces \
    ros-${ROS_DISTRO}-ros-gz-sim \
    ros-${ROS_DISTRO}-ros2bag \
    ros-${ROS_DISTRO}-rosbag2-storage-default-plugins \
    ros-${ROS_DISTRO}-sensor-msgs \
    ros-${ROS_DISTRO}-sensor-msgs-py \
    ros-${ROS_DISTRO}-std-msgs \
    ros-${ROS_DISTRO}-tf2 \
    ros-${ROS_DISTRO}-tf2-eigen \
    ros-${ROS_DISTRO}-tf2-geometry-msgs \
    ros-${ROS_DISTRO}-tf2-ros \
    ros-${ROS_DISTRO}-visualization-msgs \
    python3-matplotlib \
    python3-pandas; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/*

RUN rosdep init 2>/dev/null || true

# Clone and install MicroXRCE-DDS-Agent
RUN git clone --depth 1 https://github.com/eProsima/Micro-XRCE-DDS-Agent.git /tmp/Micro-XRCE-DDS-Agent \
    && cd /tmp/Micro-XRCE-DDS-Agent \
    && mkdir build && cd build \
    && cmake .. -DCMAKE_BUILD_TYPE=Release \
    && make -j"$(nproc)" \
    && make install \
    && ldconfig \
    && rm -rf /tmp/Micro-XRCE-DDS-Agent

# Install Python requirements for RL pipeline and PX4
RUN python3 -m venv --system-site-packages /opt/drone_rl_env && \
    /opt/drone_rl_env/bin/pip install --upgrade pip && \
    /opt/drone_rl_env/bin/pip install --no-cache-dir \
    "stable-baselines3>=2.8.0" \
    "gymnasium>=1.2.0" \
    "numpy>=1.26.0,<2.0" \
    opencv-python==4.10.0.84 \
    "rich>=15.0.0" \
    "PyYAML>=6.0" \
    "torch>=2.1.0" \
    colcon-common-extensions \
    pytest \
    argcomplete \
    cerberus \
    coverage \
    "empy==3.3.4" \
    future \
    "jinja2>=2.8" \
    jsonschema \
    kconfiglib \
    lark \
    lxml \
    "matplotlib>=3.0" \
    nunavut \
    packaging \
    "pandas>=0.21" \
    pkgconfig \
    psutil \
    pycryptodome \
    pygments \
    pymavlink \
    pynacl \
    pyros-genmsg \
    pyserial \
    "pyulog>=0.5.0" \
    pyyaml \
    requests \
    "six>=1.12.0" \
    "sympy>=1.10.1" \
    toml

# Set up user and environment
ENV USER=root
ENV HOME=/root
ENV PATH="/opt/drone_rl_env/bin:${PATH}"
ENV UAV_REPO=/workspace/obstacle_avoidance
ENV EXTERNAL_DIR=/workspace/obstacle_avoidance/external
ENV PX4_ROOT=/workspace/obstacle_avoidance/external/PX4-Autopilot
ENV ROS2_WS=/workspace/obstacle_avoidance/external/ros2_ws
ENV OPENVINS_WS=/workspace/obstacle_avoidance/external/openvins_ws
ENV PYTHONPATH=/workspace
ENV STATE_ESTIMATOR_SOURCE=openvins
ENV OPENVINS_CONFIG_PATH=/workspace/obstacle_avoidance/configs/openvins/estimator_config.yaml
ENV OPENVINS_ODOM_TOPIC=/ov_msckf/odomimu
ENV OPENVINS_IMU_TOPIC=/openvins/imu
ENV OPENVINS_CAM0_TOPIC=/openvins/cam0/image_raw
ENV OPENVINS_CAM1_TOPIC=/openvins/cam1/image_raw
ENV OPENVINS_GZ_CAM0_TOPIC=/openvins/cam0/image_raw
ENV OPENVINS_GZ_CAM1_TOPIC=/openvins/cam1/image_raw
ENV OPENVINS_USE_STEREO=1
ENV OPENVINS_MAX_CAMERAS=2
ENV OPENVINS_PUBLISH_RATE_HZ=30
ENV OPENVINS_IMU_STAMP_SOURCE=px4
ENV OPENVINS_TIMEOUT_S=0.25
ENV OPENVINS_RESET_PRIME_SOURCE=gazebo
ENV STATE_ESTIMATOR_ALLOW_GT_FALLBACK=0
ENV OPENVINS_FALLBACK_TO_GAZEBO_VO=0
WORKDIR /workspace/obstacle_avoidance

# Copy entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Expose noVNC port
EXPOSE 6080

ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]
