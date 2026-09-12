import csv
import json
import math
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


CSV_FIELDS = [
    "timestamp_ns",
    "image_timestamp_ns",
    "scan_timestamp_ns",
    "scan_time_offset_ms",
    "front_image_path",
    "scan_npz_path",
    "motor_angle",
    "motor_speed",
    "mission_label",
    "dataset_profile",
    "session_id",
]


PROFILE_ALLOWED_LABELS = {
    "drive": ["general_drive", "lane_drive", "hill_drive", "shortcut", "recovery"],
    "cone": ["cone_drive", "recovery"],
    "overtake": ["vehicle_overtake", "overtake_start", "overtake_end", "recovery"],
}


def expand_path(path: str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(path))).resolve()


def sanitize_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    return cleaned.strip("._-") or "session"


NUMBERED_SESSION_RE = re.compile(r"^(?P<base>.+?)_(?P<index>\d+)$")


def make_timestamp_session_id(session_name: str, now: Optional[datetime] = None) -> str:
    stamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{sanitize_name(session_name)}"


def session_base_name(session_name: str) -> str:
    base = sanitize_name(session_name)
    match = NUMBERED_SESSION_RE.match(base)
    if match:
        return sanitize_name(match.group("base"))
    return base


def make_next_session_id(profile_dir: Path, session_name: str) -> str:
    base = session_base_name(session_name)
    highest = 0
    pattern = re.compile(rf"^{re.escape(base)}_(\d+)$")
    if profile_dir.is_dir():
        for child in profile_dir.iterdir():
            if not child.is_dir():
                continue
            match = pattern.match(child.name)
            if match:
                highest = max(highest, int(match.group(1)))
    return f"{base}_{highest + 1:02d}"


def make_session_dir(
    output_root: str,
    dataset_profile: str,
    session_name: str,
    now: Optional[datetime] = None,
    auto_increment: bool = True,
) -> Tuple[str, Path]:
    profile = sanitize_name(dataset_profile)
    profile_dir = expand_path(output_root) / profile
    if auto_increment:
        session_id = make_next_session_id(profile_dir, session_name)
    else:
        session_id = make_timestamp_session_id(session_name, now=now)
    session_dir = profile_dir / session_id
    for relative in [
        "images/front",
        "scan",
        "debug",
    ]:
        (session_dir / relative).mkdir(parents=True, exist_ok=True)
    return session_id, session_dir


def parse_allowed_labels(value: Any, dataset_profile: str) -> List[str]:
    if value is None:
        labels: List[str] = []
    elif isinstance(value, str):
        labels = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, Iterable):
        labels = [str(part).strip() for part in value if str(part).strip()]
    else:
        labels = []
    if labels:
        return labels
    return list(PROFILE_ALLOWED_LABELS.get(str(dataset_profile), []))


def open_samples_csv(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
    writer.writeheader()
    return handle, writer


def atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=str(path.parent), delete=False
    ) as tmp:
        json.dump(data, tmp, indent=2, ensure_ascii=False, sort_keys=True)
        tmp.write("\n")
        tmp_name = tmp.name
    os.replace(tmp_name, path)


def relative_to_session(session_dir: Path, path: Optional[Path]) -> str:
    if path is None:
        return ""
    try:
        return str(path.relative_to(session_dir)).replace("\\", "/")
    except ValueError:
        return str(path)


def stamp_to_ns(msg: Any, fallback_ns: Optional[int] = None) -> int:
    header = getattr(msg, "header", None)
    stamp = getattr(header, "stamp", None)
    sec = getattr(stamp, "sec", None)
    nanosec = getattr(stamp, "nanosec", None)
    if sec is not None and nanosec is not None and (sec != 0 or nanosec != 0):
        return int(sec) * 1_000_000_000 + int(nanosec)
    if fallback_ns is not None:
        return int(fallback_ns)
    return int(datetime.now().timestamp() * 1_000_000_000)


def ros_time_to_ns(clock_now: Any) -> int:
    return int(clock_now.nanoseconds)


def _coerce_float_pair(angle_value: Any, speed_value: Any) -> Tuple[Optional[float], Optional[float], str]:
    try:
        angle = float(angle_value)
        speed = float(speed_value)
    except (TypeError, ValueError) as exc:
        return None, None, f"motor command values are not numeric: {exc}"
    if not math.isfinite(angle) or not math.isfinite(speed):
        return None, None, "motor command contains non-finite angle or speed"
    return angle, speed, "ok"


def extract_motor_command(msg: Any) -> Tuple[Optional[float], Optional[float], str]:
    if msg is None:
        return None, None, "motor message is None"

    if hasattr(msg, "angle") and hasattr(msg, "speed"):
        return _coerce_float_pair(msg.angle, msg.speed)

    data = getattr(msg, "data", None)
    if data is not None:
        try:
            if len(data) < 2:
                return None, None, "Float32MultiArray motor command needs data[0]=angle and data[1]=speed"
            return _coerce_float_pair(data[0], data[1])
        except TypeError:
            try:
                angle = float(data)
            except (TypeError, ValueError) as exc:
                return None, None, f"Float32 motor command is not numeric: {exc}"
            if not math.isfinite(angle):
                return None, None, "Float32 motor command angle is non-finite"
            return angle, None, "Float32 motor command has angle only; speed is unavailable"

    return None, None, f"unsupported motor message type: {type(msg)!r}"


def motor_from_msg(msg: Any):
    angle, speed, reason = extract_motor_command(msg)
    if angle is None or speed is None:
        raise ValueError(reason)
    return angle, speed


def string_from_msg(msg: Any, default: str = "idle") -> str:
    value = getattr(msg, "data", None)
    if value is None:
        return default
    return str(value) or default


def image_suffix(image_format: str) -> str:
    value = str(image_format).lower().strip(".")
    if value not in {"jpg", "jpeg", "png"}:
        raise ValueError("image_format must be jpg or png")
    return "jpg" if value == "jpeg" else value


def write_session_readme(
    session_dir: Path,
    session_id: str,
    dataset_profile: str,
    allowed_labels: List[str],
    topics: Dict[str, str],
) -> None:
    lines = [
        f"# IL data session: {session_id}",
        "",
        "이 폴더는 `il_data_tools`의 `il_common_recorder`가 생성한 데이터 세션입니다.",
        "",
        "## Profile",
        "",
        f"- dataset_profile: `{dataset_profile}`",
        f"- allowed_labels: `{', '.join(allowed_labels) if allowed_labels else '(none)'}`",
        "",
        "## Topics",
    ]
    for key, value in sorted(topics.items()):
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Safety",
            "",
            "이 recorder는 `/xycar_motor`를 publish하지 않습니다.",
            "최종 차량 제어와 `/xycar_motor` publish는 rule-based 주행 코드가 담당해야 합니다.",
        ]
    )
    (session_dir / "README_session.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
