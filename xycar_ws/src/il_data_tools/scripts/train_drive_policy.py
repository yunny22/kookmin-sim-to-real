#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_policy import main as train_main


if __name__ == "__main__":
    train_main(["--policy-name", "drive", "--model-type", "resnet18_lidar"] + sys.argv[1:])
