"""External demonstration controller; never part of student policy inference."""
from __future__ import annotations

from pathlib import Path

import numpy as np


class ScriptedExpert:
    def __init__(self, ctrl, cycle_start, cycle_len, dt=.02, stabilizer=(1., .04, -1., .25),
                 time_scale=1., amplitude=1., blend_steps=40):
        self.ctrl = np.asarray(ctrl, dtype=np.float32)
        # Search-tuned variants of the reference: slower/faster playback and leg swing scaled about
        # the cycle mean (blended in over blend_steps). Defaults reproduce the original gait exactly.
        self.time_scale, self.amplitude, self.blend_steps = float(time_scale), float(amplitude), int(blend_steps)
        if not (np.isfinite(self.time_scale) and .3 <= self.time_scale <= 1.5
                and np.isfinite(self.amplitude) and .3 <= self.amplitude <= 1.5 and self.blend_steps > 0):
            raise ValueError("Expert time_scale/amplitude out of range")
        self.cycle_start, self.cycle_len = int(cycle_start), int(cycle_len)
        self.dt = float(dt)
        self.stabilizer = tuple(float(value) for value in stabilizer)
        if (self.ctrl.ndim != 2 or self.ctrl.shape[1] != 19 or not np.isfinite(self.ctrl).all()
                or self.dt != .02 or not 0 <= self.cycle_start < len(self.ctrl)
                or not 0 < self.cycle_len <= len(self.ctrl) - self.cycle_start
                or len(self.stabilizer) != 4 or not np.isfinite(self.stabilizer).all()):
            raise ValueError("Malformed 50 Hz external expert reference/controller")
        self.previous_roll = self.filtered_rate = None

    @classmethod
    def from_npz(cls, path, **kwargs):
        with np.load(Path(path).resolve(strict=True), allow_pickle=False) as source:
            return cls(source["ctrl"].copy(), int(source["cycle_start"]), int(source["cycle_len"]),
                       float(source["dt"]), **kwargs)

    @staticmethod
    def roll(rotation):
        matrix = np.asarray(rotation, dtype=np.float32).reshape(-1, 3, 3)
        if not np.isfinite(matrix).all():
            raise ValueError("Nonfinite external expert pre-action rotation")
        return np.arcsin(np.clip(matrix[:, 0, 2], -1., 1.))

    def reset(self, rotation):
        self.previous_roll = self.roll(rotation).copy()
        self.filtered_rate = np.zeros_like(self.previous_roll)

    def targets(self, step, rotation):
        if self.previous_roll is None or step < 0:
            raise ValueError("Reset the external expert before generating commands")
        roll = self.roll(rotation)
        if roll.shape != self.previous_roll.shape:
            raise ValueError("Expert batch size changed during a rollout")
        self.filtered_rate = .8 * self.filtered_rate + .2 * (roll - self.previous_roll) / self.dt
        self.previous_roll = roll.copy()
        kp, kd, hip_sign, limit = self.stabilizer
        correction = np.clip(kp * roll + kd * self.filtered_rate, -limit, limit)
        targets = np.broadcast_to(self.reference(step), (len(roll), 19)).copy()
        targets[:, 0] += hip_sign * correction
        targets[:, 5] += hip_sign * correction
        targets[:, 4] += correction
        targets[:, 9] -= correction
        return targets

    def reference(self, step):
        """Reference motor targets at a control step, interpolated between 50 Hz rows."""
        start, length = self.cycle_start, self.cycle_len
        if step < start:
            return self.ctrl[step]
        phase = round((step - start) * self.time_scale, 6)
        lower = int(np.floor(phase))
        fraction = phase - lower
        first = start + lower % length
        second = start + (lower + 1) % length
        row = (1. - fraction) * self.ctrl[first] + fraction * self.ctrl[second]
        if self.amplitude != 1.:
            mean = self.ctrl[start:start + length, :10].mean(0)
            gain = 1. + (self.amplitude - 1.) * min(1., (step - start) / self.blend_steps)
            row = row.copy()
            row[:10] = mean + gain * (row[:10] - mean)
        return row
