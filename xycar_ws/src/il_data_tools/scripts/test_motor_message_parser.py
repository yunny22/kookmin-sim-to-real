#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import sys


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from il_data_tools.record_schema import extract_motor_command


class XycarMotorLike:
    angle = 12.5
    speed = 7.0


class Float32MultiArrayLike:
    def __init__(self, data):
        self.data = data


class Unknown:
    pass


def check(name, msg, expected_angle, expected_speed, reason_contains=""):
    angle, speed, reason = extract_motor_command(msg)
    ok = angle == expected_angle and speed == expected_speed
    if reason_contains:
        ok = ok and reason_contains in reason
    if ok:
        print(f"PASS {name}")
        return True
    print(
        f"FAIL {name}: got angle={angle!r} speed={speed!r} reason={reason!r}",
        file=sys.stderr,
    )
    return False


def main() -> int:
    checks = [
        check("xycar-like", XycarMotorLike(), 12.5, 7.0),
        check("float32-multi-array", Float32MultiArrayLike([3.0, 4.0]), 3.0, 4.0),
        check(
            "float32-multi-array-short",
            Float32MultiArrayLike([3.0]),
            None,
            None,
            "needs data[0]=angle",
        ),
        check("unknown", Unknown(), None, None, "unsupported motor message type"),
    ]
    return 0 if all(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
