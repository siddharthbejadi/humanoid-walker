# Third-party notices

The MIT license in `LICENSE` covers the code written for this project. The items below keep their own terms.

| Component | Where | License | Status |
|---|---|---|---|
| LaProp optimizer (Copyright (c) 2020 Wang, T. Zhikang), https://github.com/Z-T-WANG/LaProp-Optimizer | `r2dreamer/optim/laprop.py` (notice kept in the file) | MIT | verified from the file header |
| Dreamer-family model code (RSSM, networks, trainer, distributions) | `r2dreamer/*.py` | **unverified** | adapted from an open-source Dreamer implementation; upstream name and license must be confirmed and listed here before a public release |
| Robot CAD meshes and scene XML | `r2dreamer/envs/mjx_walking/assets/` | **unverified** | produced from the robot vendor's design; confirm redistribution rights or remove the meshes |

Runtime dependencies (MuJoCo, JAX, PyTorch, Hydra, ...) are installed from `requirements-lock.txt`, not bundled, and keep their own licenses.
