# syntax=docker/dockerfile:1.7

ARG ISAACSIM_VERSION=5.1.0
FROM nvcr.io/nvidia/isaac-sim:${ISAACSIM_VERSION}

ARG DEBIAN_FRONTEND=noninteractive
ARG USER_ID=1000
ARG GROUP_ID=1000

ENV ACCEPT_EULA=Y \
    OMNI_KIT_ALLOW_ROOT=1 \
    ISAACSIM_PATH=/isaac-sim \
    ISAACLAB_PATH=/workspace/isaaclab \
    HOME=/home/simuser \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    XDG_CACHE_HOME=/home/simuser/.cache \
    WANDB_DIR=/workspace/simtoolreal/outputs/wandb \
    WANDB_ARTIFACT_DIR=/workspace/simtoolreal/outputs/wandb_artifacts \
    WANDB_CACHE_DIR=/workspace/simtoolreal/outputs/wandb_cache \
    WANDB_DATA_DIR=/workspace/simtoolreal/outputs/wandb_data

SHELL ["/bin/bash", "-c"]

USER root

RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    git \
    libglib2.0-0 \
    ncurses-term \
    wget \
    && apt-get -y autoremove \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY --from=isaaclab / ${ISAACLAB_PATH}
RUN chmod +x ${ISAACLAB_PATH}/isaaclab.sh \
    && ln -sf ${ISAACSIM_PATH} ${ISAACLAB_PATH}/_isaac_sim

RUN ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install toml
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/var/cache/apt \
    ${ISAACLAB_PATH}/isaaclab.sh -p ${ISAACLAB_PATH}/tools/install_deps.py apt ${ISAACLAB_PATH}/source \
    && apt-get -y autoremove \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
RUN --mount=type=cache,target=/root/.cache/pip \
    ${ISAACLAB_PATH}/isaaclab.sh --install

WORKDIR /workspace/simtoolreal
COPY . /workspace/simtoolreal

RUN --mount=type=cache,target=/root/.cache/pip \
    ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install \
    imageio \
    "imageio[ffmpeg]" \
    tyro \
    wandb \
    && ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install -e /workspace/simtoolreal/src/isaaclab_env

RUN sed -i 's/2\.4\.31/2.4.30/g' \
        ${ISAACLAB_PATH}/apps/isaaclab.python.kit \
        ${ISAACLAB_PATH}/source/isaaclab/isaaclab/sim/converters/urdf_converter.py \
    && mkdir -p \
        /workspace/simtoolreal/outputs \
        /home/simuser/.cache/ov \
        /home/simuser/.config \
        /home/simuser/.local/share/ov \
        /home/simuser/.nv \
        ${ISAACSIM_PATH}/kit/cache \
        ${ISAACSIM_PATH}/kit/data \
        ${ISAACSIM_PATH}/kit/logs \
        /tmp/matplotlib \
    && chown -R ${USER_ID}:${GROUP_ID} /workspace/simtoolreal /home/simuser /tmp/matplotlib \
    && chmod o+rx ${ISAACSIM_PATH} \
    && chmod -R a+rwx ${ISAACSIM_PATH}/kit/cache ${ISAACSIM_PATH}/kit/data ${ISAACSIM_PATH}/kit/logs

USER ${USER_ID}:${GROUP_ID}
RUN git config --global --add safe.directory /workspace/simtoolreal

ENV PYTHONPATH=/workspace/simtoolreal/src/isaaclab_env:/workspace/simtoolreal

CMD ["/bin/bash"]
