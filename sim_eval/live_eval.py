#!/usr/bin/env python3
"""Interactive real-time MolmoAct2 control of a single ManiSkill episode.

The policy server and this process are intentionally separate. Start the server
first, then run this module. Type a new instruction at any time; it is applied
at the next control tick without stopping the simulation.

Example:
  uv run python -m sim_eval.live_eval \
    --policy-type remote-yam --remote-url http://127.0.0.1:8202/act \
    --env-id BimanualYAMPutEverythingInBox-v1
"""

from __future__ import annotations

import argparse
import queue
import threading
import time
from pathlib import Path

import gymnasium as gym
import numpy as np

import mani_skill.envs  # noqa: F401
from .tasks import *  # noqa: F401,F403
from .inference.client import DroidClient, YAMClient
from .run_eval import DEFAULT_LANGUAGE_INSTRUCTIONS, _capture_frame


def _stdin_reader(commands: queue.Queue[str]) -> None:
    print("\nType an instruction and press Enter (':reset', ':quit', ':save <path>').", flush=True)
    while True:
        try:
            line = input("instruction> ").strip()
        except (EOFError, KeyboardInterrupt):
            commands.put(":quit")
            return
        if line:
            commands.put(line)
        if line == ":quit":
            return


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy-type", choices=["remote-droid", "remote-yam"], default="remote-yam")
    p.add_argument("--remote-url", required=True)
    p.add_argument("--env-id", default="BimanualYAMPutEverythingInBox-v1")
    p.add_argument("--instruction", default=None)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--n-action-steps", type=int, default=1)
    p.add_argument("--request-timeout", type=float, default=60.0)
    p.add_argument("--control-freq", type=int, default=30)
    p.add_argument("--sim-freq", type=int, default=150)
    p.add_argument("--output-dir", default="sim_eval/outputs/live")
    p.add_argument("--record-video", action="store_true", help="Capture frames for :save; disabled by default for headless reliability")
    args = p.parse_args()

    instruction = args.instruction or DEFAULT_LANGUAGE_INSTRUCTIONS.get(args.env_id, "perform the task")
    client_cls = DroidClient if args.policy_type == "remote-droid" else YAMClient
    client = client_cls(args.remote_url, n_action_steps=args.n_action_steps, request_timeout=args.request_timeout)

    env = gym.make(
        args.env_id, obs_mode="rgb", control_mode="pd_joint_pos", render_mode="rgb_array",
        max_episode_steps=args.max_steps, reward_mode="none",
        sensor_configs=dict(shader_pack="minimal"),
        sim_config=dict(sim_freq=args.sim_freq, control_freq=args.control_freq),
        sim_backend="physx_cpu", render_backend="cpu",
    )
    commands: queue.Queue[str] = queue.Queue()
    threading.Thread(target=_stdin_reader, args=(commands,), daemon=True).start()
    frames = []
    obs, _ = env.reset(seed=42)
    step = 0
    started = time.monotonic()

    try:
        while step < args.max_steps:
            while True:
                try:
                    command = commands.get_nowait()
                except queue.Empty:
                    break
                if command == ":quit":
                    return
                if command == ":reset":
                    obs, _ = env.reset(seed=42 + step + 1)
                    client.reset(); step = 0; frames.clear()
                    print("[live] episode reset", flush=True)
                elif command.startswith(":save"):
                    target = command.partition(" ")[2].strip() or f"{args.output_dir}/rollout.mp4"
                    if frames:
                        import imageio.v2 as imageio
                        Path(target).parent.mkdir(parents=True, exist_ok=True)
                        imageio.mimsave(target, frames, fps=args.control_freq)
                        print(f"[live] saved {target}", flush=True)
                else:
                    instruction = command
                    client.reset()
                    print(f"[live] instruction = {instruction!r}", flush=True)

            t0 = time.monotonic()
            action = client.infer(obs, instruction)
            obs, reward, terminated, truncated, info = env.step(action)
            if args.record_video:
                frame = _capture_frame(env)
                if frame is not None:
                    frames.append(frame)
            success = info.get("success", False)
            success = bool(np.asarray(success).any())
            qpos = np.asarray(obs["agent"]["qpos"]).reshape(-1)
            print(
                f"[step {step:04d}] instruction={instruction!r} action={np.array2string(action, precision=3)} "
                f"qpos={np.array2string(qpos, precision=3)} in_box={info.get('n_in_box')} "
                f"success={success} policy_ms={(time.monotonic()-t0)*1000:.0f}",
                flush=True,
            )
            step += 1
            if bool(np.asarray(terminated).any()) or bool(np.asarray(truncated).any()):
                print("[live] episode finished; type :reset or a new instruction", flush=True)
                client.reset()
                while True:
                    time.sleep(0.1)
                    try:
                        command = commands.get_nowait()
                    except queue.Empty:
                        continue
                    if command == ":reset":
                        obs, _ = env.reset(seed=42 + step + 1); step = 0; frames.clear(); break
                    if command == ":quit": return
                    instruction = command; obs, _ = env.reset(seed=42 + step + 1); step = 0; frames.clear(); break
            else:
                # Keep the sim paced at the configured control frequency when inference is faster.
                time.sleep(max(0.0, 1.0 / args.control_freq - (time.monotonic() - t0)))
    finally:
        env.close()
        print(f"[live] stopped after {step} steps ({time.monotonic()-started:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
