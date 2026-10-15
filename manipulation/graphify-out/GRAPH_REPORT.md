# Graph Report - manipulation  (2026-10-15)

## Corpus Check
- 60 files · ~140,766 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 9 file(s) not represented in the graph (top: .patch 3, (none) 2, .bat 2)

## Summary
- 308 nodes · 698 edges · 9 communities (7 shown, 2 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 33 edges (avg confidence: 0.87)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Pose gate, simulator and IK
- Tests, filter and smoothing
- Firmware: servo link and camera task
- Gripper network and training
- Hand experiment and MoveNet fine-tune
- Real-arm driver and jog tool
- Firmware: calibration console
- Figures and dashboard

## God Nodes (most connected - your core abstractions)
1. `arm_vectors()` - 17 edges
2. `Humanoid` - 16 edges
3. `RightArm` - 15 edges
4. `ArmIK3` - 15 edges
5. `Execute()` - 15 edges
6. `OneEuro` - 14 edges
7. `retarget_sequence()` - 12 edges
8. `forward_fractions()` - 11 edges
9. `Print()` - 11 edges
10. `MoveNet` - 10 edges

## Surprising Connections (you probably didn't know these)
- `Chain` --uses--> `MoveNet`  [INFERRED]
  demo.py → armcopy/pose.py
- `Geometry` --uses--> `Humanoid`  [INFERRED]
  tests/test_core.py → armcopy/sim.py
- `Chain` --uses--> `OneEuro`  [INFERRED]
  demo.py → armcopy/filters.py
- `Filter` --uses--> `OneEuro`  [INFERRED]
  tests/test_core.py → armcopy/filters.py
- `RobotDriver` --uses--> `RightArm`  [INFERRED]
  tests/test_core.py → armcopy/robot.py

## Import Cycles
- None detected.

## Communities (9 total, 2 thin omitted)

### Community 0 - "Pose gate, simulator and IK"
Cohesion: 0.08
Nodes (9): MoveNet, ArmIK, Humanoid, humanoid_angles(), run(), summarize(), main(), blur_head() (+1 more)

### Community 1 - "Tests, filter and smoothing"
Cohesion: 0.07
Nodes (16): OneEuro, hand_crop(), skin_fraction(), _warp(), arm_vectors(), forward_fractions(), ArmIK3, Chain (+8 more)

### Community 2 - "Firmware: servo link and camera task"
Cohesion: 0.07
Nodes (23): HAL_I2C_MspInit(), i2c1_timing_100khz(), ImuStream_Set(), walker_thread_fct(), ServoCal_GetLimits(), ServoCal_JointCommanded(), ServoCal_JointUnused(), ServoCal_NoteCommand() (+15 more)

### Community 3 - "Gripper network and training"
Cohesion: 0.06
Nodes (10): HandNet, Reader, augment(), predict(), report(), predict(), report(), evaluate() (+2 more)

### Community 4 - "Hand experiment and MoveNet fine-tune"
Cohesion: 0.09
Nodes (16): build_model(), _chain(), set_pose(), fill_short_gaps(), lin(), retarget(), smooth(), targets_from_landmarks() (+8 more)

### Community 5 - "Real-arm driver and jog tool"
Cohesion: 0.10
Nodes (3): load_setup(), RightArm, RobotDriver

### Community 6 - "Firmware: calibration console"
Cohesion: 0.23
Nodes (16): CommandPulse(), DisableBoard(), DisableChannel(), Execute(), ExportCsv(), FindJoint(), InitBoard(), MoveSlowly() (+8 more)

## Knowledge Gaps
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `RightArm` connect `Real-arm driver and jog tool` to `Tests, filter and smoothing`?**
  _High betweenness centrality (0.051) - this node is a cross-community bridge._
- **Should `Pose gate, simulator and IK` be split into smaller, more focused modules?**
  _Cohesion score 0.07683000604960677 - nodes in this community are weakly interconnected._
- **Why does `ArmIK3` connect `Tests, filter and smoothing` to `Pose gate, simulator and IK`?**
  _High betweenness centrality (0.028) - this node is a cross-community bridge._
- **Should `Tests, filter and smoothing` be split into smaller, more focused modules?**
  _Cohesion score 0.06531986531986532 - nodes in this community are weakly interconnected._
- **Why does `arm_vectors()` connect `Tests, filter and smoothing` to `Pose gate, simulator and IK`, `Hand experiment and MoveNet fine-tune`?**
  _High betweenness centrality (0.027) - this node is a cross-community bridge._
- **Should `Firmware: servo link and camera task` be split into smaller, more focused modules?**
  _Cohesion score 0.0700354609929078 - nodes in this community are weakly interconnected._
- **Should `Gripper network and training` be split into smaller, more focused modules?**
  _Cohesion score 0.06236786469344609 - nodes in this community are weakly interconnected._