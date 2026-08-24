#!/usr/bin/env python3
"""Capture one RealSense image per rokae_assembly_real camera config."""

from __future__ import annotations

import argparse
import ast
import sys
import time
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from franka_env.camera.rs_capture import RSCapture


DEFAULT_CONFIG = Path("examples/experiments/rokae_assembly_real/config.py")
DEFAULT_OUTPUT_DIR = Path("outputs/captured_images")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture RealSense frames using EnvConfig.REALSENSE_CAMERAS resolution settings."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--warmup-frames", type=int, default=15)
    return parser.parse_args()


def load_camera_config(config_path: Path) -> dict:
    config_path = Path(config_path)
    tree = ast.parse(config_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "EnvConfig":
            for stmt in node.body:
                if not isinstance(stmt, ast.Assign):
                    continue
                if any(
                    isinstance(target, ast.Name) and target.id == "REALSENSE_CAMERAS"
                    for target in stmt.targets
                ):
                    return ast.literal_eval(stmt.value)
    raise RuntimeError(f"Could not find EnvConfig.REALSENSE_CAMERAS in {config_path}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cameras = load_camera_config(args.config)

    for name, kwargs in cameras.items():
        serial = kwargs["serial_number"]
        dim = kwargs.get("dim", (640, 480))
        fps = kwargs.get("fps", 15)
        exposure = kwargs.get("exposure", 40000)
        print(f"Opening {name}: serial={serial}, dim={dim}, fps={fps}, exposure={exposure}")

        cap = RSCapture(
            name=name,
            serial_number=serial,
            dim=dim,
            fps=fps,
            exposure=exposure,
        )
        try:
            image = None
            for _ in range(max(1, args.warmup_frames)):
                ok, image = cap.read()
                if not ok:
                    image = None
                time.sleep(0.02)
            if image is None:
                raise RuntimeError(f"Failed to capture image from {name} ({serial})")

            output_path = args.output_dir / f"{name}_{serial}_{dim[0]}x{dim[1]}.png"
            cv2.imwrite(str(output_path), image)
            print(f"Saved {output_path} shape={image.shape}")
        finally:
            cap.close()


if __name__ == "__main__":
    main()
