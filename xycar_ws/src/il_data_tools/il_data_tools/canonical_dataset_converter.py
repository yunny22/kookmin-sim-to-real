from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import time
from pathlib import Path

import cv2

from il_data_tools.canonical_artifacts import (
    CanonicalVisibilityAugmenter,
    canonical_lane_observation_present,
)


def load_visibility(path_value: str) -> dict[str, float]:
    defaults = {
        "white_both_probability": 0.462,
        "white_one_probability": 0.512,
        "white_none_probability": 0.026,
        "yellow_visible_probability": 0.571,
    }
    if not path_value:
        return defaults
    path = Path(path_value).expanduser().resolve()
    with path.open(encoding="utf-8") as handle:
        configured = json.load(handle).get("recommended_sim_visibility", {})
    for key in defaults:
        if key in configured:
            defaults[key] = float(configured[key])
    return defaults


def convert_session(
    source: Path,
    output_root: Path,
    visibility: dict[str, float],
    seed: int,
    variant: str = "v1",
) -> Path:
    source = source.expanduser().resolve()
    csv_path = source / "samples.csv"
    if not csv_path.is_file():
        raise RuntimeError(f"samples.csv missing: {source}")
    safe_variant = "".join(
        character for character in str(variant) if character.isalnum() or character in "-_"
    )
    if not safe_variant:
        raise ValueError("variant must contain a file-name-safe character")
    destination = output_root / f"{source.name}_real_visibility_{safe_variant}"
    if destination.is_dir() and (destination / "samples.csv").is_file():
        with csv_path.open(newline="", encoding="utf-8-sig") as handle:
            source_rows = sum(1 for _ in handle) - 1
        metadata_path = destination / "metadata.json"
        metadata = (
            json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata_path.is_file()
            else {}
        )
        if (
            int(metadata.get("source_rows", -1)) == source_rows
            and int(metadata.get("seed", -1)) == int(seed)
            and metadata.get("variant") == safe_variant
            and metadata.get("visibility_profile") == visibility
        ):
            print(f"reuse converted session: {destination}")
            return destination

    staging = destination.with_name(f".{destination.name}.staging-{os.getpid()}")
    if staging.exists():
        shutil.rmtree(staging)
    image_dir = staging / "images" / "front"
    image_dir.mkdir(parents=True)
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    augmenter = CanonicalVisibilityAugmenter(
        seed=seed,
        white_probabilities=(
            visibility["white_both_probability"],
            visibility["white_one_probability"],
            visibility["white_none_probability"],
        ),
        yellow_visible_probability=visibility["yellow_visible_probability"],
        prevent_blank=True,
    )
    output_rows = []
    skipped_blank_canonical = 0
    for index, row in enumerate(rows, start=1):
        source_image = Path(row["front_image_path"]).expanduser()
        if not source_image.is_absolute():
            source_image = source / source_image
        image = cv2.imread(str(source_image), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"failed to read canonical image: {source_image}")
        if not canonical_lane_observation_present(image):
            skipped_blank_canonical += 1
            continue
        output, _, _ = augmenter.process(image)
        suffix = source_image.suffix.lower() if source_image.suffix else ".png"
        if suffix not in {".png", ".jpg", ".jpeg"}:
            suffix = ".png"
        output_name = f"{row.get('image_timestamp_ns') or row.get('timestamp_ns')}{suffix}"
        output_path = image_dir / output_name
        if not cv2.imwrite(str(output_path), output):
            raise RuntimeError(f"failed to write canonical image: {output_path}")
        converted = dict(row)
        converted["front_image_path"] = str(output_path.relative_to(staging))
        scan_value = (row.get("scan_npz_path") or "").strip()
        if scan_value:
            scan_path = Path(scan_value).expanduser()
            if not scan_path.is_absolute():
                scan_path = (source / scan_path).resolve()
            converted["scan_npz_path"] = str(scan_path)
        converted["session_id"] = row.get("session_id") or source.name
        output_rows.append(converted)
        if index % 5000 == 0:
            print(f"{source.name}: converted {index}/{len(rows)}", flush=True)

    with (staging / "samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    source_metadata = {}
    metadata_path = source / "metadata.json"
    if metadata_path.is_file():
        source_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata = {
        "created_at_unix": time.time(),
        "derived_from": str(source),
        "source_rows": len(rows),
        "rows": len(output_rows),
        "skipped_blank_canonical": skipped_blank_canonical,
        "seed": int(seed),
        "variant": safe_variant,
        "session_id_policy": "preserved_from_source_to_prevent_split_leakage",
        "visibility_profile": visibility,
        "source_metadata": source_metadata,
    }
    (staging / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if destination.exists():
        shutil.rmtree(destination)
    staging.rename(destination)
    print(f"converted session: {destination}")
    return destination


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-dir", action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--real-reference-profile", default="")
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--variant", default="v1")
    args = parser.parse_args(argv)
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    visibility = load_visibility(args.real_reference_profile)
    outputs = [
        convert_session(
            Path(value), output_root, visibility, args.seed + index, args.variant
        )
        for index, value in enumerate(args.session_dir)
    ]
    print(json.dumps({"sessions": [str(path) for path in outputs]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
