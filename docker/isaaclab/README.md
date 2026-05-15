# Isaac Lab Docker

5080-targeted Docker runtime for the Isaac Lab direct environment and CleanRL trainer.

```bash
cd /home/waybaba/code/simtoolreal
./scripts/docker/build_isaaclab.sh
```

Run a shell:

```bash
./scripts/docker/run_isaaclab_5080.sh
```

Run the simplified fixed-hammer training:

```bash
./scripts/docker/run_isaaclab_5080.sh ./scripts/docker/run_isaaclab_simple_hammer_5080.sh
```

Run with a custom env count:

```bash
NUM_ENVS=8190 ./scripts/docker/run_isaaclab_5080.sh ./scripts/docker/run_isaaclab_simple_hammer_5080.sh
```

Run detached for a serious training job:

```bash
RUN_NAME=isaaclab_simple_hammer_5080_full_$(date +%Y%m%d_%H%M%S)
DETACH=true NAME=simtoolreal-isaaclab-full RUN_NAME="$RUN_NAME" NUM_ENVS=8190 \
  ./scripts/docker/run_isaaclab_5080.sh ./scripts/docker/run_isaaclab_simple_hammer_5080.sh
tail -f "outputs/${RUN_NAME}.log"
```

Stop a detached run:

```bash
docker stop simtoolreal-isaaclab-full
```

Outputs are bind-mounted to `outputs/` in the project root.
