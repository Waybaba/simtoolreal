# syntax=docker/dockerfile:1.7

FROM nvcr.io/nvidia/pytorch:21.09-py3

ARG DEBIAN_FRONTEND=noninteractive
ARG USER_ID=1000
ARG GROUP_ID=1000

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=all \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    XDG_CACHE_HOME=/tmp/.cache

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    curl \
    git \
    git-lfs \
    make \
    cmake \
    gcc \
    g++ \
    ninja-build \
    pkg-config \
    zip \
    unzip \
    pigz \
    libegl1 \
    libgl1 \
    libglvnd0 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libxcursor-dev \
    libxrandr-dev \
    libxinerama-dev \
    libxi-dev \
    mesa-common-dev \
    mesa-vulkan-drivers \
    vulkan-tools \
    && rm -rf /var/lib/apt/lists/*

# Match NVIDIA's Isaac Gym container workaround: prefer NVIDIA EGL/Vulkan over Mesa.
RUN rm -f \
    /usr/lib/x86_64-linux-gnu/libEGL_mesa.so.0 \
    /usr/lib/x86_64-linux-gnu/libEGL_mesa.so.0.0.0 \
    /usr/share/glvnd/egl_vendor.d/50_mesa.json

COPY docker/isaacgym/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json
COPY docker/isaacgym/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json
RUN chmod 0644 \
    /usr/share/vulkan/icd.d/nvidia_icd.json \
    /usr/share/glvnd/egl_vendor.d/10_nvidia.json \
    && rm -f \
    /usr/share/vulkan/icd.d/intel_icd*.json \
    /usr/share/vulkan/icd.d/lvp_icd*.json \
    /usr/share/vulkan/icd.d/radeon_icd*.json

RUN python -m pip install \
    "pip==23.3.2" \
    "setuptools==59.8.0" \
    "wheel==0.41.3"

RUN python -m pip install --no-build-isolation \
    cloudpickle==3.1.2 \
    gym==0.23.1 \
    hydra-core==1.3.2 \
    imageio==2.35.1 \
    "imageio[ffmpeg]" \
    jinja2 \
    matplotlib==3.7.5 \
    networkx==2.2 \
    numpy==1.23.0 \
    omegaconf==2.3.0 \
    opencv-python-headless==4.5.5.64 \
    pillow==10.4.0 \
    protobuf==3.20.3 \
    "psutil>=5.9.0" \
    pyopenssl \
    pyrender \
    pytorch3d==0.3.0 \
    pysdf==0.1.9 \
    pyvirtualdisplay \
    "pyyaml>=6.0" \
    requests \
    scipy==1.10.1 \
    "setproctitle>=1.2.2" \
    tensorboard==2.14.0 \
    tensorboardx==2.6.2.2 \
    termcolor \
    transforms3d==0.4.2 \
    trimesh==3.23.5 \
    tyro==1.0.13 \
    urdfpy==0.0.22 \
    wandb==0.12.21 \
    warp-lang==0.10.1 \
    yourdfpy==0.0.60

COPY --from=isaacgym / /opt/isaacgym
RUN python -m pip install --no-deps -e /opt/isaacgym/python

WORKDIR /workspace/simtoolreal
COPY . /workspace/simtoolreal
RUN python -m pip install --no-deps -e /workspace/simtoolreal/rl_games \
    && python -m pip install --no-deps -e /workspace/simtoolreal

RUN groupadd --gid ${GROUP_ID} simuser \
    && useradd --uid ${USER_ID} --gid ${GROUP_ID} --create-home --shell /bin/bash simuser \
    && mkdir -p /workspace/simtoolreal/outputs /tmp/matplotlib /tmp/.cache \
    && chown -R simuser:simuser /workspace /tmp/matplotlib /tmp/.cache

USER simuser

ENV PYTHONPATH=/workspace/simtoolreal:/workspace/simtoolreal/rl_games \
    WANDB_DIR=/workspace/simtoolreal/outputs/wandb

CMD ["/bin/bash"]
