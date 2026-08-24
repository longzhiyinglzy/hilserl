#!/usr/bin/env python3
"""Interactively select IMAGE_CROP regions for Rokae RealSense views."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import cv2


DEFAULT_IMAGE_DIR = Path("/home/user/code/hil-serl/outputs/captured_images")
DEFAULT_CONFIG = Path("examples/experiments/rokae_assembly_real/config.py")
DEFAULT_VIEWS = ("side", "wrist1", "wrist2")
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select crop ROIs for side/wrist1/wrist2 images and print an "
            "IMAGE_CROP snippet for config.py."
        )
    )
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--views", nargs="+", default=list(DEFAULT_VIEWS))
    parser.add_argument(
        "--image",
        action="append",
        default=[],
        metavar="VIEW=PATH",
        help="Override one image path, for example --image side=/tmp/side.png",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Display scale for ROI selection; coordinates are converted back.",
    )
    parser.add_argument(
        "--no-config-dim",
        action="store_true",
        help=(
            "Keep crop coordinates in the captured image resolution instead of "
            "mapping them to the dim values in config.py."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_IMAGE_DIR / "roi_crops.json",
        help="Where to save selected crop coordinates.",
    )
    parser.add_argument(
        "--preview-dir",
        type=Path,
        default=DEFAULT_IMAGE_DIR / "roi_preview",
        help="Where to save cropped preview images.",
    )
    return parser.parse_args()


def load_camera_config(config_path: Path) -> dict[str, dict[str, object]]:
    tree = ast.parse(config_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "EnvConfig":
            for stmt in node.body:
                if not isinstance(stmt, ast.Assign):
                    continue
                if not any(
                    isinstance(target, ast.Name) and target.id == "REALSENSE_CAMERAS"
                    for target in stmt.targets
                ):
                    continue
                cameras = ast.literal_eval(stmt.value)
                return {
                    key: {
                        "serial_number": str(value["serial_number"]),
                        "dim": tuple(value["dim"]) if "dim" in value else None,
                    }
                    for key, value in cameras.items()
                    if isinstance(value, dict) and "serial_number" in value
                }
    return {}


def parse_image_overrides(overrides: list[str]) -> dict[str, Path]:
    parsed = {}
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid --image override {item!r}; expected VIEW=PATH.")
        view, path = item.split("=", 1)
        parsed[view.strip()] = Path(path).expanduser()
    return parsed


def list_images(image_dir: Path) -> list[Path]:
    images = [
        path
        for path in sorted(image_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    ]
    if not images:
        raise FileNotFoundError(f"No images found in {image_dir}")
    return images


def choose_image_for_view(view: str, candidates: list[Path]) -> Path:
    print(f"\nCould not auto-match image for {view!r}. Choose one:")
    for idx, path in enumerate(candidates, start=1):
        print(f"  {idx}. {path}")
    while True:
        choice = input(f"Image number for {view}: ").strip()
        try:
            index = int(choice)
        except ValueError:
            print("Please enter a number.")
            continue
        if 1 <= index <= len(candidates):
            return candidates[index - 1]
        print("Choice out of range.")


def build_image_map(
    views: list[str],
    image_dir: Path,
    config_path: Path,
    overrides: dict[str, Path],
) -> dict[str, Path]:
    camera_config = load_camera_config(config_path)
    candidates = list_images(image_dir)
    image_map = {}

    for view in views:
        if view in overrides:
            image_map[view] = overrides[view]
            continue

        serial = camera_config.get(view, {}).get("serial_number")
        matched = []
        if serial is not None:
            matched = [path for path in candidates if serial in path.name]
        if not matched:
            matched = [path for path in candidates if view.lower() in path.name.lower()]

        image_map[view] = matched[0] if len(matched) == 1 else choose_image_for_view(view, candidates)

    return image_map


def map_crop_to_target_dim(
    crop: dict[str, int],
    source_width: int,
    source_height: int,
    target_dim: tuple[int, int] | None,
) -> dict[str, int]:
    target_width, target_height = (
        (source_width, source_height) if target_dim is None else target_dim
    )
    if target_dim is None or target_dim == (source_width, source_height):
        return {
            **crop,
            "target_width": target_width,
            "target_height": target_height,
            "mapped_from_source": False,
        }

    x_scale = target_width / source_width
    y_scale = target_height / source_height
    return {
        **crop,
        "x1": round(crop["x1"] * x_scale),
        "y1": round(crop["y1"] * y_scale),
        "x2": round(crop["x2"] * x_scale),
        "y2": round(crop["y2"] * y_scale),
        "width": round(crop["width"] * x_scale),
        "height": round(crop["height"] * y_scale),
        "source_width": source_width,
        "source_height": source_height,
        "target_width": target_width,
        "target_height": target_height,
        "mapped_from_source": True,
    }


def select_roi(
    view: str,
    image_path: Path,
    scale: float,
    target_dim: tuple[int, int] | None,
) -> tuple[dict[str, int], object]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")

    display = image
    if scale != 1.0:
        if scale <= 0:
            raise ValueError("--scale must be greater than 0.")
        display = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    window = f"Select ROI: {view} ({image_path.name})"
    print(f"\nSelecting {view}: {image_path}")
    print("Drag ROI, then press ENTER or SPACE. Press c to cancel and use full image.")
    x, y, w, h = cv2.selectROI(window, display, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(window)

    if w == 0 or h == 0:
        height, width = image.shape[:2]
        x, y, w, h = 0, 0, width, height
    elif scale != 1.0:
        x = round(x / scale)
        y = round(y / scale)
        w = round(w / scale)
        h = round(h / scale)

    height, width = image.shape[:2]
    x1 = max(0, min(int(x), width - 1))
    y1 = max(0, min(int(y), height - 1))
    x2 = max(x1 + 1, min(int(x + w), width))
    y2 = max(y1 + 1, min(int(y + h), height))
    crop = image[y1:y2, x1:x2]

    source_crop = {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "width": x2 - x1,
        "height": y2 - y1,
        "source": str(image_path),
        "source_width": width,
        "source_height": height,
    }
    return map_crop_to_target_dim(source_crop, width, height, target_dim), crop


def format_config_snippet(crops: dict[str, dict[str, int]]) -> str:
    lines = ["IMAGE_CROP = {"]
    for view, crop in crops.items():
        lines.append(
            f'    "{view}": lambda img: img[{crop["y1"]}:{crop["y2"]}, '
            f'{crop["x1"]}:{crop["x2"]}],'
        )
    lines.append("}")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    image_dir = args.image_dir.expanduser()
    config_path = args.config.expanduser()
    camera_config = load_camera_config(config_path)
    image_map = build_image_map(
        views=args.views,
        image_dir=image_dir,
        config_path=config_path,
        overrides=parse_image_overrides(args.image),
    )

    args.preview_dir.mkdir(parents=True, exist_ok=True)
    crops = {}
    for view in args.views:
        target_dim = None if args.no_config_dim else camera_config.get(view, {}).get("dim")
        crop, cropped_image = select_roi(view, image_map[view], args.scale, target_dim)
        crops[view] = crop
        preview_path = args.preview_dir / f"{view}_crop.png"
        cv2.imwrite(str(preview_path), cropped_image)
        print(
            f"{view}: x={crop['x1']}:{crop['x2']}, y={crop['y1']}:{crop['y2']} "
            f"-> {preview_path}"
        )
        if crop.get("mapped_from_source"):
            print(
                f"  mapped from {crop['source_width']}x{crop['source_height']} "
                f"to config dim {crop['target_width']}x{crop['target_height']}"
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(crops, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    snippet = format_config_snippet(crops)
    snippet_path = args.out.with_suffix(".py")
    snippet_path.write_text(snippet + "\n", encoding="utf-8")

    print("\nSaved ROI json:", args.out)
    print("Saved config snippet:", snippet_path)
    print("\nPaste this into EnvConfig:")
    print(snippet)


if __name__ == "__main__":
    main()
