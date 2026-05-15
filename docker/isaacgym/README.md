# Isaac Gym Docker

3090-only Docker runtime for the existing Isaac Gym training path. The wrapper defaults to physical GPU 0, half the CPU cores, half the system memory, and video off for throughput.

```bash
cd /home/waybaba/code/simtoolreal
./scripts/docker/build_isaacgym.sh
```

Run a shell:

```bash
./scripts/docker/run_isaacgym_3090.sh
```

Run training:

```bash
./scripts/docker/run_isaacgym_3090.sh ./scripts/run_isaacgym_simple_marker_3090.sh
```

Run with a custom env count:

```bash
NUM_ENVS=4092 ./scripts/docker/run_isaacgym_3090.sh ./scripts/run_isaacgym_simple_marker_3090.sh
```

Run with camera video enabled:

```bash
CAPTURE_VIDEO=true NUM_ENVS=4092 ./scripts/docker/run_isaacgym_3090.sh ./scripts/run_isaacgym_simple_marker_3090.sh
```

Override isolation:

```bash
CPUSET_CPUS=0-3 MEMORY=24g ./scripts/docker/run_isaacgym_3090.sh
```

Note: the RTX 5080 does not work with this Isaac Gym stack because the official Isaac Gym container base uses an old CUDA/PyTorch stack without `sm_120` kernels, while Isaac Gym Preview 4 only ships Python 3.6-3.8 bindings.
