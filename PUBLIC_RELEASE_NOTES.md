# Public Release Notes

## Source Selection

This staging release is extracted from the locally available remote-tracking
reference:

- Repository: `yunny22/kookmin_sim_to_real`
- Reference: `origin/simulation`
- Commit: `7b9016d04dbda8fd0ed754c4da754d781751c1ad`
- Commit date/message: 2026-08-05, `Add speed-25 straight RL and continuous curve control`

The source was read through a local Git object store. No source workspace,
branch, ref, or Git history was modified. This staging directory has no GitHub
remote and has not been pushed.

## Source Differences

The comparison used:

- Workspace A: `xycar_ws/src/{il_data_tools,xycar_rl,xycar_perception}` at
  `678522e`
- History repository B: `origin/simulation` at `7b9016d`

The `il_data_tools` and `xycar_perception` package trees are identical at
the compared revisions. The `xycar_rl` tree differs: the shared workspace
ends at the July 24 RL state, while `origin/simulation` includes later RL
source changes through August 5. The local `simulation` branch contains a
later Rule-mission commit, but its three selected package trees match the
remote-tracking source. For a reproducible three-package snapshot, this release
uses `7b9016d` and does not use the mixed `빠킹` checkout.

This is a curated code release, not a claim that every later RL variant was a
competition-ready result.

## Included

- `il_data_tools`: canonical preprocessing, recorder/session schema,
  dataset conversion, domain randomization, BC training helpers, policy
  inference, and relevant launches/tests
- `xycar_rl`: Gazebo environment interface, replay/transition handling,
  camera-speed BC and TD3+BC, policy runtime, rollout, evaluation, rewards,
  termination, launches, config, and tests
- `xycar_perception`: canonical road representation, lane tracker, optional
  model-backed lane adapter, perception node, simulation-oriented config,
  launches, and tests
- concise package READMEs and root documentation

## Excluded

The following were intentionally not extracted:

- all trained weights and exported model artifacts, including `.pt`, `.pth`,
  `.onnx`, and `.engine` files
- raw datasets, captured images/scans, transition files, rosbag data, and
  runtime recordings
- real-vehicle dynamics records and the vehicle-specific real-reference profile
- fisheye camera calibration and real-track perception configuration
- Gazebo world assets, vehicle bridge, external YOLO source/runtime, hardware
  drivers, Rule-based Final source, parking source, SLAM source, KAI source,
  and ROS build products

## Sanitization

- Added a public-focused `.gitignore` for artifacts, recordings, datasets,
  caches, and local calibration.
- Replaced maintainer names/emails with neutral project-team metadata.
- Removed the missing `collect_parallel_rule_expert` packaging entry point;
  the referenced module does not exist in the selected source snapshot.
- Updated the generic camera-perception launch so an omitted real-camera
  calibration must be supplied explicitly instead of defaulting to a
  vehicle-specific file.
- Added model and calibration scope notes in place of the excluded artifacts.

No production result was changed or fabricated. Model and dataset paths remain
runtime inputs.

## Provenance

- `il_data_tools`: PERSONAL authorship evidence in the selected history,
  pending model/data publication-right confirmation
- `xycar_rl`: PERSONAL authorship evidence in the selected history, but
  project integration and result claims remain TEAM
- `xycar_perception`: CO-DEVELOPED; the selected history includes more than
  one contributor
- overall project: CO-DEVELOPED / TEAM

Commit attribution alone does not establish publication rights for all modules,
models, datasets, calibration, or competition outcomes.

## Validation

The staging release was checked without copying models, datasets, calibration,
worlds, or vehicle records:

- Python syntax, package XML, package structure, packaging entry points, and
  local README links passed.
- `colcon build --packages-select il_data_tools xycar_perception xycar_rl`
  passed for all three packages.
- Launch-description generation passed for nine core simulation and runtime
  launches after sourcing ROS 2 and the staged package paths.
- The source-only unit-test subset passed: 87 tests, with one real-camera-profile
  test intentionally deselected. Three further tests require the omitted Gazebo
  world. `test_centerline_fit` additionally requires the external
  `kaiev26_msgs` interface.

A successful package build does not exercise Gazebo closed-loop driving or real
vehicle control. Those require external bridge/world assets, reviewed model
weights, data, and hardware configuration.

## Remaining Issues Before GitHub Publication

1. Review the full source history for previously committed secrets, models,
   datasets, raw calibration, and machine paths.
2. Confirm rights to every model, dataset derivative, real-vehicle record,
   route/world asset, and competition media item.
3. Install the exact companion ROS packages, Gazebo world/bridge, and Python
   dependencies, then run the simulation and real-handoff launch paths with
   sanitized local artifacts.
4. Review whether the post-July 24 RL variants should be kept, split, or
   further reduced before public publication.
6. Create the GitHub repository and push only after these checks are complete.
