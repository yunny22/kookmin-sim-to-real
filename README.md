# Kookmin Final – Reinforcement Learning & Sim-to-Real

This repository is provided for portfolio and research demonstration purposes.
No separate open-source reuse license is granted; third-party dependencies keep
their own terms.

This is a cleaned public release of Taeyun Kim's individual Sim-to-Real
research track, developed in the Team KAI context of the Kookmin University
autonomous-driving competition final. It explores whether a Gazebo-based
learning pipeline can improve the high-speed control-stability limits observed
after the preliminary Imitation Learning project. It contains source code only:
trained weights, datasets, raw vehicle captures, and private calibration
records are deliberately excluded.

## Overview

The project progressed from canonical perception and a Behavioral Cloning (BC)
baseline to offline TD3+BC, then attempted to transfer the policy from Gazebo
to the real vehicle. The source provides the closed-loop environment,
data/transition tooling, policy runtime, and perception components used to
study that transfer. It does not claim that Reinforcement Learning solved the
high-speed oscillation problem.

## Motivation

The preliminary learned driver showed steering oscillation and control
stability limits as speed increased. Repeated real-vehicle RL training was not
practical, so the project built a Gazebo training and evaluation workflow and
then examined how much of the learned behavior could be transferred to the
vehicle.

## System Architecture

```text
Gazebo or vehicle camera
        ↓
Canonical road representation
        ↓
BC policy → temporal TD3+BC policy
        ↓
Steering and speed commands
        ↓
Gazebo closed-loop evaluation / real-vehicle handoff

Gazebo environment: image, optional LiDAR, odometry, reward and termination
```

`xycar_perception` produces the canonical road image. The camera-speed actor
uses the current canonical RGB frame or a concatenation of two consecutive
frames, then predicts normalized steering and speed actions. The Gazebo
environment also receives optional LiDAR and odometry for synchronization,
episode control, reward, and termination handling.

## Canonical Representation

The perception package transforms the road observation into a fixed canonical
road representation rather than training directly on raw camera appearance.
The pipeline includes canonical conversion, lane tracking, visibility-style
artifact injection, and domain-randomization utilities. This was intended to
reduce the visual gap between simulation and the real camera; it did not make
the physical behavior of the two systems identical.

## Behavioral Cloning Baseline

`train_camera_speed_bc.py` trains the camera-speed BC actor from recorded
transitions. The BC actor is then used as the initial policy for the offline RL
workflow. Data collection, session validation, preprocessing, dataset
conversion, and runtime inference are kept in `il_data_tools`.

## Offline RL – TD3+BC

`train_camera_speed_td3_bc.py` and `td3_bc.py` implement the offline TD3+BC
workflow. The policy supports one or two image frames; the two-frame option
retains recent visual context. The actor outputs steering and speed, while the
twin critics evaluate an action from the canonical image input. Reward,
termination, replay, transition, and rollout code are retained with the
training source.

## Gazebo Training / Evaluation

`gazebo_env.py` implements a Gym-style environment around the Gazebo Xycar
interface. It synchronizes camera, optional LiDAR, and odometry observations,
applies actions, and evaluates progress, lane departure, collision, and other
termination conditions. `evaluate_closed_loop.py` and the simulation launch
source are included for closed-loop evaluation with a compatible external
Gazebo world and bridge.

## Sim-to-Real Calibration

The project measured sensor delay, control delay, and vehicle behavior, then
used those observations to adjust simulation parameters and the transfer
pipeline. The raw measurement records and camera calibration files are not
included here. Low-speed real-vehicle application was checked, but at higher
speed the remaining gap from friction, tire-road interaction, and vehicle
behavior could not be reproduced closely enough for stable deployment.

## Individual Project

This project was independently designed, implemented, and evaluated by Taeyun
Kim. The research pipeline includes canonical perception and camera
preprocessing, dataset and transition preparation, Behavioral Cloning, the
camera-speed policy, offline TD3+BC training, temporal-input experiments,
Gazebo closed-loop evaluation, real-vehicle sensor/control-delay and vehicle
behavior measurement, simulation calibration, Sim-to-Real transfer, and
limitation analysis.

The Kookmin Final competition was a Team KAI activity. This individual research
track does not claim ownership of the separate team competition system or
third-party dependencies.

## Results

- Gazebo closed-loop driving and evaluation were implemented.
- The learned policy was applied to the real vehicle at low speed.
- High-speed Sim-to-Real stability was not sufficiently secured within the
  competition schedule.

No sample count, MAE, seed-completion, or competition-lap metrics are reported
as public headline results in this release.

## Limitations

Visual canonicalization reduced part of the perception gap, but it could not
fully reproduce sensor/control timing, friction, tire-road interaction, or
high-speed vehicle behavior. The final competition driving system therefore
moved toward a separate Rule-based architecture. That source is intentionally
not included in this repository.

## Insight

Sim-to-Real performance depended on both the perception gap and the physical
fidelity of the simulation. Sensor/control delay and vehicle behavior were
measured and used for calibration, but unreproduced friction and real vehicle
dynamics still left a high-speed gap.

## Repository Structure

```text
.
├── README.md
├── PUBLIC_RELEASE_NOTES.md
└── xycar_ws/src/
    ├── il_data_tools/       # canonical data, preprocessing, BC runtime
    ├── xycar_perception/    # canonical road perception
    └── xycar_rl/            # Gazebo environment, TD3+BC, evaluation
```

The original ROS workspace layout is retained so package-relative imports and
launch paths stay recognizable. Models are represented only by explanatory
`models/README.md` files.

## Environment

ROS 2 기반 Gazebo 환경에서 개발·검증했다. 실행에는 각 package manifest에
명시된 Python 및 외부 Xycar message/bridge 의존성이 필요하다. Gazebo world,
vehicle bridge, 외부 lane-model runtime, hardware driver, model weight,
dataset, 실차 camera calibration은 이 공개본에 포함하지 않는다.

## How to Run

Build the three packages in a ROS 2 workspace after installing their external
dependencies:

```bash
cd /path/to/kookmin-sim-to-real/xycar_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install \
  --packages-select il_data_tools xycar_perception xycar_rl
source install/setup.bash
```

Training, evaluation, or deployment additionally requires publication-cleared
model weights, a compatible dataset or transition directory, and the external
Gazebo/vehicle interface. Pass artifact paths through the scripts or ROS launch
arguments; no public default model is supplied. Full closed-loop reproduction
also requires the competition world and bridge, which are not part of this
curated release.

### Source-only checks

The source-only test subset can be run after sourcing ROS 2 and the three
package directories. Tests that require the excluded Gazebo world or
vehicle-specific real-camera profile are intentionally left out of this check.

```bash
source /opt/ros/humble/setup.bash
export PYTHONPATH="$PWD/xycar_ws/src/il_data_tools:$PWD/xycar_ws/src/xycar_rl:$PWD/xycar_ws/src/xycar_perception:$PYTHONPATH"
python3 -m pytest -q \
  xycar_ws/src/il_data_tools/test/test_canonical_artifacts.py \
  xycar_ws/src/il_data_tools/test/test_canonical_dataset_converter.py \
  xycar_ws/src/il_data_tools/test/test_canonical_pipeline.py \
  xycar_ws/src/il_data_tools/test/test_reliability.py \
  xycar_ws/src/xycar_perception/test/test_canonical_road.py \
  xycar_ws/src/xycar_perception/test/test_canonical_lane_tracker.py \
  xycar_ws/src/xycar_perception/test/test_yolo_canonical.py \
  xycar_ws/src/xycar_rl/test/test_reward_termination.py \
  xycar_ws/src/xycar_rl/test/test_replay_residual.py \
  xycar_ws/src/xycar_rl/test/test_sampling.py \
  xycar_ws/src/xycar_rl/test/test_rollout_policy.py \
  -k 'not test_real_config_matches_measured_lane_pipeline'
```

`test_centerline_fit.py` also needs the external `kaiev26_msgs` interface and
is therefore an integration check rather than a source-only test in this
release.

## Context / Credits

- Individual research project: Taeyun Kim
- Competition context: Team KAI / Kookmin University autonomous-driving competition
- Cross-check history remote: [`yunny22/kookmin_sim_to_real`](https://github.com/yunny22/kookmin_sim_to_real)
- Public source revision: `origin/simulation` at `7b9016d`
- Public release organization: Taeyun Kim; see
  [`PUBLIC_RELEASE_NOTES.md`](PUBLIC_RELEASE_NOTES.md)

Models, data, calibration and media are excluded from this source-only
portfolio release.
