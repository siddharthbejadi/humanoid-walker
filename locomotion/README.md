# Locomotion: a script-free walking policy

This project is the original content of this repository: a recurrent R2Dreamer policy, trained by behaviour cloning of a scripted gait, that walks the 19-servo humanoid in
MuJoCo / MJX-Warp from 45 proprioceptive numbers at 50 Hz, with no gait table or reference at run time. **Status: simulation only** (see the honest limitations in the main README).

**Nothing was moved or edited to make this folder.** The locomotion files stay exactly where they were; this page is an index:

| What | Where |
|---|---|
| Full write-up: results, how it works, limitations | [main README, Locomotion section](../README.md#locomotion) |
| Policy, simulator, training code | [`../r2dreamer`](../r2dreamer) |
| Results, figures, media | [`../results`](../results) · [`../figures`](../figures) · [`../media`](../media) |
| Deployment (laptop to robot, test protocol, contract) | [`../deploy`](../deploy) |
| Tools, scripts, notes | [`../tools`](../tools) · [`../scripts`](../scripts) · [`../docs`](../docs) |
| Checkpoints and Hugging Face weights | [`../checkpoints`](../checkpoints) |

<p align="center"><img src="../media/walk_1100.gif" width="360" alt="Trained student walking in simulation"></p>

The [manipulation project](../manipulation) reuses this project's robot model, meshes and servo link (`deploy/serial_servo_link.py`).
