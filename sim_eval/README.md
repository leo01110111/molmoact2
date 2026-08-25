# sim_eval — MolmoAct2 Simulation Evaluation

Zero-shot evaluation of [MolmoAct2](https://huggingface.co/allenai/MolmoAct2-DROID) policies inside [ManiSkill](https://github.com/haosulab/ManiSkill) simulation.

## Directory layout

```
sim_eval/
├── run_eval.py          # CLI entry point
├── inference/
│   ├── client.py        # DroidClient / YAMClient (HTTP ↔ /act)
│   └── common.py        # Schemas, state/action adapters, obs helpers
├── robots/
│   ├── franka_droid.py  # Franka FR3 + Robotiq gripper (DROID)
│   └── bimanual_yam.py  # Bimanual YAM arms (YAM)
├── tasks/
│   ├── droid_tasks/
│   │   └── droid_put_everything_in_box.py
│   └── yam_tasks/
│       └── bimanual_put_everything_in_box.py
├── assets/              # Robot meshes / URDFs
└── scripts/
    └── download_assets.py
```

## Setup

**1. Install dependencies**

```bash
uv sync          # from repo root
```

**2. Download robot assets**

Assets (URDF meshes for Franka, MJCF files for YAM) are not committed to the repo.
Download them once:

```bash
uv run python sim_eval/scripts/download_assets.py
```

This pulls from `TreeePlanter/molmoact2-sim-eval-assets` on HuggingFace and places the
files under `sim_eval/assets/`.  Pass `--force` to re-download.

## Running evaluation

Start the inference server as described in the [main README](../README.md), then run the evaluator:

```bash
# YAM
uv run python -m sim_eval.run_eval \
    --policy-type remote-yam \
    --remote-url http://<host>:8202/act \
    -e BimanualYAMPutEverythingInBox-v1

# DROID
uv run python -m sim_eval.run_eval \
    --policy-type remote-droid \
    --remote-url http://<host>:8000/act \
    -e DroidPutEverythingInBox-v1
```

Results are written to `sim_eval/outputs/<timestamp>/results.json`.
Videos and per-episode camera frames are saved alongside.

### Interactive live evaluation

Start the local YAM policy server (the server can load the downloaded checkpoint
directly), then start the interactive simulator:

```bash
uv run python examples/yam/host_server_yam.py \
  --host 127.0.0.1 --port 8202 \
  --repo-id .models/MolmoAct2-BimanualYAM \
  --device cuda:0 --dtype bfloat16

uv run python -m sim_eval.live_eval \
  --policy-type remote-yam \
  --remote-url http://127.0.0.1:8202/act \
  --env-id BimanualYAMPutEverythingInBox-v1
```

The terminal accepts a new language instruction at any time. `:reset` starts a
new seeded episode, `:save <path>` writes a recorded rollout when
`--record-video` is enabled, and `:quit` exits. Each control tick prints the
instruction, action, simulator joint state, object count in the box, success
flag, and policy latency.

For a desktop interface with a live simulator camera, text entry, reset, stop,
task selection, adjustable episode limit, and status log, use the Tk GUI:

```bash
DISPLAY=:1 MUJOCO_GL=egl uv run python -m sim_eval.gui_live_eval \
  --policy-type remote-yam \
  --remote-url http://127.0.0.1:8202/act \
  --env-id BimanualYAMPutEverythingInBox-v1
```

The GUI task menu currently includes both `BimanualYAMPutEverythingInBox-v1`
and `DroidPutEverythingInBox-v1`. Selecting a task rebuilds the local simulator
and switches the matching action client. If the two policies use different
servers, provide both endpoints; otherwise `--remote-url` is used for either:

```bash
DISPLAY=:0 MUJOCO_GL=egl uv run python -m sim_eval.gui_live_eval \
  --remote-url http://100.102.154.75:8202/act \
  --yam-url http://100.102.154.75:8202/act \
  --droid-url http://100.102.154.75:8000/act
```

The `Episode limit` control applies a new step budget immediately. The GUI
process owns the simulator and rendering; the policy server only performs
model inference.

### Laptop viewer with compute-host inference

This split lets the compute host keep the GPU model while the laptop owns the
interactive simulator window. Both machines must be on the same Tailscale
network. On the compute host, bind the policy server to its Tailscale address:

```bash
TAILSCALE_ADDR=$(tailscale ip -4)
uv run python examples/yam/host_server_yam.py \
  --host "$TAILSCALE_ADDR" --port 8202 \
  --repo-id .models/MolmoAct2-BimanualYAM \
  --device cuda:0 --dtype bfloat16
```

On the laptop, clone this repository, run `uv sync` and the asset download
command above, then launch the GUI with the compute host’s Tailscale IP:

```bash
DISPLAY=:0 MUJOCO_GL=egl uv run python -m sim_eval.gui_live_eval \
  --remote-url http://100.102.154.75:8202/act \
  --env-id BimanualYAMPutEverythingInBox-v1 \
  --max-steps 2000
```

The laptop needs the simulator dependencies and assets, but not the model
checkpoint. The endpoint must be reachable from the laptop, for example with
`curl http://100.102.154.75:8202/health` if a health route is enabled.

### Key flags

| Flag | Default | Description |
|------|---------|-------------|
| `--policy-type` / `-p` | `remote-yam` | `remote-droid` or `remote-yam` |
| `--remote-url` | — | Full `/act` endpoint URL (required) |
| `-e` | — | One or more ManiSkill env IDs |
| `-n` | `10` | Episodes per task |
| `--max-episode-steps` | `800` | Step limit per episode |
| `--language-instruction` | per-env default | Language instruction override |
| `--n-action-steps` | full chunk | Actions to execute per server call |
| `--save-video` | `True` | Save rollout videos |

## Available environments

| Env ID | Robot | Task |
|--------|-------|------|
| `DroidPutEverythingInBox-v1` | Franka FR3 + Robotiq | Pick lego duplo + tennis ball → box |
| `BimanualYAMPutEverythingInBox-v1` | Bimanual YAM | Pick lego duplo + tennis ball → box |

## Adding a new task

1. Create `sim_eval/tasks/<embodiment>_tasks/my_task.py` — subclass `BaseEnv`, register
   with `@register_env`, import the robot from `...robots.<robot>` (side-effect: registers
   the agent with ManiSkill).
2. Add an import in the parent `__init__.py` so `from ..tasks import *` picks it up.
3. Add an entry in `DEFAULT_LANGUAGE_INSTRUCTIONS` in `run_eval.py`.
