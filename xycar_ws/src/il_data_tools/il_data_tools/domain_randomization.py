from __future__ import annotations

import hashlib
import json
import math
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, Tuple

import yaml


PRESETS = {
    "baseline": {
        "visual": 0.0,
        "sensor": 0.0,
        "dynamics": 0.0,
        "ambient_range": (0.75, 0.75),
    },
    "visual_light": {
        "visual": 0.65,
        "sensor": 0.15,
        "dynamics": 0.0,
        "ambient_range": (0.78, 0.98),
    },
    "visual_dark": {
        "visual": 0.75,
        "sensor": 0.25,
        "dynamics": 0.0,
        "ambient_range": (0.38, 0.62),
    },
    "sensor": {
        "visual": 0.20,
        "sensor": 1.0,
        "dynamics": 0.15,
        "ambient_range": (0.62, 0.86),
    },
    "dynamics": {
        "visual": 0.15,
        "sensor": 0.15,
        "dynamics": 1.0,
        "ambient_range": (0.65, 0.88),
    },
    "mixed": {
        "visual": 1.0,
        "sensor": 0.75,
        "dynamics": 0.75,
        "ambient_range": (0.42, 0.96),
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _uniform(rng: random.Random, center: float, radius: float) -> float:
    return center + rng.uniform(-radius, radius)


def _rgba_text(rgb: Iterable[float]) -> str:
    values = [min(1.0, max(0.0, float(value))) for value in rgb]
    return " ".join(f"{value:.6f}" for value in [*values, 1.0])


def _parse_numbers(element: ET.Element, count: int) -> list[float]:
    values = [float(part) for part in (element.text or "").split()]
    if len(values) < count:
        values.extend([0.0] * (count - len(values)))
    return values


def _set_color(element: ET.Element, rgb: Tuple[float, float, float]) -> None:
    element.text = _rgba_text(rgb)


def _randomized_color(
    rng: random.Random,
    base: Tuple[float, float, float],
    magnitude: float,
) -> Tuple[float, float, float]:
    brightness = rng.uniform(0.72, 1.18) if magnitude else 1.0
    return tuple(
        min(1.0, max(0.015, value * brightness + rng.uniform(-0.08, 0.08) * magnitude))
        for value in base
    )


def _find_model(world: ET.Element, name: str) -> ET.Element:
    for model in world.findall("model"):
        if model.get("name") == name:
            return model
    raise ValueError(f"model not found in world: {name}")


def _randomize_room(
    world: ET.Element,
    rng: random.Random,
    magnitude: float,
) -> Dict[str, object]:
    moved = []
    recolored = 0
    movable_prefixes = (
        "room_table_top_",
        "room_chair_back_",
        "room_chair_seat_",
        "room_orange_cone_reference",
        "room_black_display_stand",
    )
    for model in world.findall("model"):
        name = model.get("name", "")
        if not name.startswith("room_"):
            continue
        for material in model.findall(".//material"):
            ambient = material.find("ambient")
            diffuse = material.find("diffuse")
            source = ambient if ambient is not None else diffuse
            if source is None:
                continue
            values = _parse_numbers(source, 4)
            color = _randomized_color(rng, tuple(values[:3]), magnitude)
            if ambient is not None:
                _set_color(ambient, color)
            if diffuse is not None:
                _set_color(diffuse, color)
            recolored += 1
        if magnitude <= 0.0 or not name.startswith(movable_prefixes):
            continue
        pose = model.find("pose")
        if pose is None:
            continue
        values = _parse_numbers(pose, 6)
        dx = rng.uniform(-0.20, 0.20) * magnitude
        dy = rng.uniform(-0.15, 0.15) * magnitude
        dyaw = math.radians(rng.uniform(-8.0, 8.0)) * magnitude
        values[0] += dx
        values[1] += dy
        values[5] += dyaw
        pose.text = " ".join(f"{value:.9f}" for value in values)
        moved.append({"name": name, "dx": dx, "dy": dy, "dyaw": dyaw})
    return {"recolored_visuals": recolored, "moved_models": moved}


def _randomize_camera_and_lidar(
    xycar: ET.Element,
    rng: random.Random,
    magnitude: float,
) -> Dict[str, object]:
    camera = xycar.find(".//sensor[@name='front_camera']")
    lidar = xycar.find(".//sensor[@name='lidar']")
    if camera is None or lidar is None:
        raise ValueError("front_camera or lidar sensor missing from Xycar model")

    camera_pose = camera.find("pose")
    if camera_pose is None:
        raise ValueError("front camera pose is missing")
    values = _parse_numbers(camera_pose, 6)
    offsets = {
        "x_m": rng.uniform(-0.005, 0.005) * magnitude,
        "y_m": rng.uniform(-0.008, 0.008) * magnitude,
        "z_m": rng.uniform(-0.010, 0.010) * magnitude,
        "pitch_rad": math.radians(rng.uniform(-1.5, 1.5)) * magnitude,
        "yaw_rad": math.radians(rng.uniform(-1.0, 1.0)) * magnitude,
    }
    values[0] += offsets["x_m"]
    values[1] += offsets["y_m"]
    values[2] += offsets["z_m"]
    values[4] += offsets["pitch_rad"]
    values[5] += offsets["yaw_rad"]
    camera_pose.text = " ".join(f"{value:.9f}" for value in values)

    camera_noise = camera.find("./camera/noise")
    lidar_noise = lidar.find("./lidar/noise")
    camera_stddev = rng.uniform(0.002, 0.012) * magnitude
    lidar_stddev = rng.uniform(0.003, 0.025) * magnitude
    for noise, stddev in ((camera_noise, camera_stddev), (lidar_noise, lidar_stddev)):
        if noise is None:
            continue
        noise.find("type").text = "gaussian" if stddev > 0.0 else "none"
        noise.find("mean").text = "0"
        noise.find("stddev").text = f"{stddev:.9f}"

    return {
        "camera_pose_offsets": offsets,
        "camera_noise_stddev": camera_stddev,
        "lidar_noise_stddev_m": lidar_stddev,
    }


def _randomize_dynamics(
    world: ET.Element,
    xycar: ET.Element,
    rng: random.Random,
    magnitude: float,
) -> Dict[str, float]:
    acceleration_scale = _uniform(rng, 1.0, 0.20 * magnitude)
    ackermann = xycar.find(".//plugin[@name='gz::sim::systems::AckermannSteering']")
    if ackermann is None:
        raise ValueError("AckermannSteering plugin missing")
    max_accel = 2.0 * acceleration_scale
    ackermann.find("max_acceleration").text = f"{max_accel:.6f}"
    ackermann.find("min_acceleration").text = f"{-max_accel:.6f}"

    friction_scale = _uniform(rng, 1.0, 0.18 * magnitude)
    track = _find_model(world, "kookmin_track")
    ode = track.find("./link[@name='floor_base']/collision/surface/friction/ode")
    bullet = track.find("./link[@name='floor_base']/collision/surface/friction/bullet")
    if ode is not None:
        for name in ("mu", "mu2"):
            child = ode.find(name)
            if child is not None:
                child.text = f"{50.0 * friction_scale:.6f}"
    if bullet is not None and bullet.find("friction") is not None:
        bullet.find("friction").text = f"{friction_scale:.6f}"
    return {
        "acceleration_scale": acceleration_scale,
        "floor_friction_scale": friction_scale,
    }


def _randomize_bridge_config(
    source_path: Path,
    output_path: Path,
    rng: random.Random,
    magnitude: float,
) -> Dict[str, float]:
    with source_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    params = data["xycar_motor_bridge"]["ros__parameters"]
    speed_gain_scale = _uniform(rng, 1.0, 0.06 * magnitude)
    steering_delay_scale = _uniform(rng, 1.0, 0.25 * magnitude)
    speed_delay_scale = _uniform(rng, 1.0, 0.25 * magnitude)
    params["speed_gain"] = float(params["speed_gain"]) * speed_gain_scale
    params["steering_delay_sec"] = max(
        0.0, float(params["steering_delay_sec"]) * steering_delay_scale
    )
    params["speed_delay_sec"] = max(
        0.0, float(params["speed_delay_sec"]) * speed_delay_scale
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)
    return {
        "speed_gain_scale": speed_gain_scale,
        "steering_delay_scale": steering_delay_scale,
        "speed_delay_scale": speed_delay_scale,
        "speed_gain_mps_per_cmd": params["speed_gain"],
        "steering_delay_sec": params["steering_delay_sec"],
        "speed_delay_sec": params["speed_delay_sec"],
    }


def generate_randomized_assets(
    source_world: str,
    output_world: str,
    source_bridge_config: str,
    output_bridge_config: str,
    manifest_path: str,
    seed: int,
    preset: str,
) -> Dict[str, object]:
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; choose from {sorted(PRESETS)}")
    source_world_path = Path(source_world).expanduser().resolve()
    output_world_path = Path(output_world).expanduser().resolve()
    source_bridge_path = Path(source_bridge_config).expanduser().resolve()
    output_bridge_path = Path(output_bridge_config).expanduser().resolve()
    manifest_output_path = Path(manifest_path).expanduser().resolve()
    if not source_world_path.is_file():
        raise FileNotFoundError(source_world_path)
    if not source_bridge_path.is_file():
        raise FileNotFoundError(source_bridge_path)

    profile = PRESETS[preset]
    rng = random.Random(int(seed))
    tree = ET.parse(source_world_path)
    root = tree.getroot()
    world = root.find("world")
    if world is None:
        raise ValueError("SDF world element is missing")
    xycar = _find_model(world, "xycar_ackermann")

    ambient_level = rng.uniform(*profile["ambient_range"])
    color_tint = [rng.uniform(-0.035, 0.035) * profile["visual"] for _ in range(3)]
    ambient_rgb = tuple(ambient_level + tint for tint in color_tint)
    background_rgb = tuple(min(1.0, value + 0.10) for value in ambient_rgb)
    scene = world.find("scene")
    if scene is None:
        raise ValueError("world scene element is missing")
    _set_color(scene.find("ambient"), ambient_rgb)
    _set_color(scene.find("background"), background_rgb)
    scene.find("shadows").text = "true" if rng.random() < 0.35 * profile["visual"] else "false"

    sun = world.find("light[@name='sun']")
    sun_intensity = rng.uniform(0.65, 1.20) if profile["visual"] else 1.0
    if sun is not None:
        if sun.find("intensity") is not None:
            sun.find("intensity").text = f"{sun_intensity:.6f}"
        if sun.find("diffuse") is not None:
            _set_color(sun.find("diffuse"), tuple(min(1.0, value * 0.92) for value in ambient_rgb))

    room = _randomize_room(world, rng, profile["visual"])
    sensors = _randomize_camera_and_lidar(xycar, rng, profile["sensor"])
    dynamics = _randomize_dynamics(world, xycar, rng, profile["dynamics"])
    bridge = _randomize_bridge_config(
        source_bridge_path,
        output_bridge_path,
        rng,
        profile["dynamics"],
    )

    output_world_path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(output_world_path, encoding="utf-8", xml_declaration=True)

    manifest = {
        "schema_version": 1,
        "seed": int(seed),
        "preset": preset,
        "source_world": str(source_world_path),
        "source_world_sha256": _sha256(source_world_path),
        "generated_world": str(output_world_path),
        "generated_bridge_config": str(output_bridge_path),
        "visual": {
            "ambient_rgb": ambient_rgb,
            "background_rgb": background_rgb,
            "sun_intensity": sun_intensity,
            **room,
        },
        "sensors": sensors,
        "dynamics": {**dynamics, **bridge},
    }
    manifest_output_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_output_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    return manifest
