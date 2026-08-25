#!/usr/bin/env python3
"""Desktop GUI for live MolmoAct2 simulation control."""

from __future__ import annotations

import argparse
import queue
import threading
import time
import tkinter as tk
from tkinter import ttk

import gymnasium as gym
import numpy as np
from PIL import Image, ImageTk

import mani_skill.envs  # noqa: F401
from .tasks import *  # noqa: F401,F403
from .inference.client import DroidClient, YAMClient
from .run_eval import DEFAULT_LANGUAGE_INSTRUCTIONS


class LiveGui:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.commands: queue.Queue[tuple[str, str | None]] = queue.Queue()
        self.updates: queue.Queue[tuple[str, object]] = queue.Queue()
        self.stop_event = threading.Event()
        self.reset_event = threading.Event()
        self.instruction = args.instruction or DEFAULT_LANGUAGE_INSTRUCTIONS.get(args.env_id, "perform the task")
        self.env = None
        self.client = None

        self.root = tk.Tk()
        self.root.title("MolmoAct2 · Live Simulation")
        self.root.geometry("1180x760")
        self.root.minsize(900, 600)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self._build_ui()
        self.worker = threading.Thread(target=self._run_simulation, daemon=True)
        self.worker.start()
        self.root.after(80, self._drain_updates)

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=3)
        self.root.columnconfigure(1, weight=2)
        self.root.rowconfigure(1, weight=1)

        header = ttk.Frame(self.root, padding=(14, 12, 14, 8))
        header.grid(row=0, column=0, columnspan=2, sticky="ew")
        ttk.Label(header, text="MolmoAct2 Live Simulation", font=("TkDefaultFont", 16, "bold")).pack(side="left")
        self.status = ttk.Label(header, text="Starting…")
        self.status.pack(side="right")

        visual = ttk.LabelFrame(self.root, text="Simulation view", padding=8)
        visual.grid(row=1, column=0, sticky="nsew", padx=(14, 7), pady=(0, 14))
        visual.rowconfigure(0, weight=1)
        visual.columnconfigure(0, weight=1)
        self.image_label = ttk.Label(visual, text="Waiting for simulator frame", anchor="center")
        self.image_label.grid(row=0, column=0, sticky="nsew")

        side = ttk.Frame(self.root, padding=(7, 0, 14, 14))
        side.grid(row=1, column=1, sticky="nsew")
        side.rowconfigure(3, weight=1)
        side.columnconfigure(0, weight=1)

        ttk.Label(side, text="Instruction").grid(row=0, column=0, sticky="w")
        entry_row = ttk.Frame(side)
        entry_row.grid(row=1, column=0, sticky="ew", pady=(4, 10))
        entry_row.columnconfigure(0, weight=1)
        self.entry = ttk.Entry(entry_row)
        self.entry.insert(0, self.instruction)
        self.entry.grid(row=0, column=0, sticky="ew")
        self.entry.bind("<Return>", lambda _event: self.send_instruction())
        ttk.Button(entry_row, text="Send", command=self.send_instruction).grid(row=0, column=1, padx=(6, 0))

        buttons = ttk.Frame(side)
        buttons.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        ttk.Button(buttons, text="Reset episode", command=lambda: self.commands.put(("reset", None))).pack(side="left")
        ttk.Button(buttons, text="Stop", command=self.close).pack(side="right")

        self.log = tk.Text(side, height=12, width=42, state="disabled", wrap="word")
        self.log.grid(row=3, column=0, sticky="nsew")
        self.log.configure(font=("TkFixedFont", 9))

    def send_instruction(self) -> None:
        text = self.entry.get().strip()
        if text:
            self.commands.put(("instruction", text))

    @staticmethod
    def _frame_from_obs(obs: dict) -> np.ndarray | None:
        sensors = obs.get("sensor_data") or {}
        data = sensors.get("base_camera") or sensors.get("top_cam")
        if not isinstance(data, dict):
            return None
        image = data.get("rgb")
        if image is None:
            return None
        if hasattr(image, "detach"):
            image = image.detach().cpu().numpy()
        image = np.asarray(image)
        if image.ndim == 4:
            image = image[0]
        if image.dtype != np.uint8:
            image = np.clip(image * 255 if image.max() <= 1 else image, 0, 255).astype(np.uint8)
        return image

    def _run_simulation(self) -> None:
        client_cls = DroidClient if self.args.policy_type == "remote-droid" else YAMClient
        self.client = client_cls(self.args.remote_url, n_action_steps=self.args.n_action_steps, request_timeout=120)
        self.env = gym.make(
            self.args.env_id, obs_mode="rgb", control_mode="pd_joint_pos", render_mode="rgb_array",
            max_episode_steps=self.args.max_steps, reward_mode="none",
            sensor_configs=dict(shader_pack="minimal"),
            sim_config=dict(sim_freq=150, control_freq=self.args.control_freq),
            sim_backend="physx_cpu", render_backend="cpu",
        )
        obs, _ = self.env.reset(seed=42)
        step = 0
        self.updates.put(("status", "Running"))
        while not self.stop_event.is_set() and step < self.args.max_steps:
            while True:
                try:
                    command, value = self.commands.get_nowait()
                except queue.Empty:
                    break
                if command == "instruction" and value:
                    self.instruction = value
                    self.client.reset()
                    self.updates.put(("log", f"instruction: {value}"))
                elif command == "reset":
                    obs, _ = self.env.reset(seed=42 + step + 1)
                    self.client.reset()
                    step = 0
                    self.updates.put(("log", "episode reset"))

            started = time.monotonic()
            action = self.client.infer(obs, self.instruction)
            obs, _reward, terminated, truncated, info = self.env.step(action)
            frame = self._frame_from_obs(obs)
            if frame is not None:
                self.updates.put(("frame", frame))
            success = bool(np.asarray(info.get("success", False)).any())
            self.updates.put(("status", f"Step {step} · {'SUCCESS' if success else 'running'}"))
            self.updates.put(("log", f"step {step:04d}  latency={(time.monotonic()-started)*1000:.0f} ms  in_box={info.get('n_in_box')}"))
            step += 1
            if bool(np.asarray(terminated).any()) or bool(np.asarray(truncated).any()):
                self.updates.put(("log", "episode finished; press Reset episode"))
                break
        if self.env is not None:
            self.env.close()

    def _drain_updates(self) -> None:
        try:
            while True:
                kind, value = self.updates.get_nowait()
                if kind == "status":
                    self.status.configure(text=str(value))
                elif kind == "log":
                    self.log.configure(state="normal")
                    self.log.insert("end", str(value) + "\n")
                    self.log.see("end")
                    self.log.configure(state="disabled")
                elif kind == "frame":
                    image = Image.fromarray(value)
                    image.thumbnail((760, 620), Image.Resampling.LANCZOS)
                    photo = ImageTk.PhotoImage(image)
                    self.image_label.configure(image=photo, text="")
                    self.image_label.image = photo
        except queue.Empty:
            pass
        if not self.stop_event.is_set():
            self.root.after(80, self._drain_updates)

    def close(self) -> None:
        if self.stop_event.is_set():
            return
        self.stop_event.set()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-type", choices=["remote-droid", "remote-yam"], default="remote-yam")
    parser.add_argument("--remote-url", required=True)
    parser.add_argument("--env-id", default="BimanualYAMPutEverythingInBox-v1")
    parser.add_argument("--instruction", default=None)
    parser.add_argument("--n-action-steps", type=int, default=10)
    parser.add_argument("--control-freq", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=2000)
    LiveGui(parser.parse_args()).run()


if __name__ == "__main__":
    main()
