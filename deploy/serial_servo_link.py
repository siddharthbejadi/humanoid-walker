"""Laptop -> STM32 servo command link (binary frames) for robot_host.py.

NOT run on hardware. The framing is unit-tested offline (python serial_servo_link.py --selftest).

Frame (little-endian), sent every 20 ms:
    0xAA 0x55 | seq:u8 | flags:u8 | 19 x u16 pulse_us | crc16:u16        = 2+1+1+38+2 = 44 bytes
    flags bit0 = RELAX (STM32 must stop driving the servos immediately; pulses are ignored)
    crc16 = CRC-16/CCITT-FALSE over bytes [seq .. last pulse]
Step budget: 44 B * 10 bit / 115200 baud = 3.8 ms of the 20 ms period (full duplex with the IMU stream).

STM32 MUST (these are safety requirements, not options):
  * validate sync + CRC, drop bad frames;
  * if no valid frame for 100 ms  ->  behave as RELAX (stop PWM / pull the PCA9685 OE pin high);
  * clamp every pulse to [min_us, max_us] from its own table, independent of the laptop;
  * accept RELAX at any time and latch it until a fresh frame with RELAX=0 arrives after a 1 s idle;
  * keep a hardware power cut (physical switch) on the servo supply.

Calibration table (JSON, YOU fill it in): list of 19 objects
    {"neutral_us": 1500, "sign": 1, "us_per_rad": 190.0, "min_us": 1200, "max_us": 1800}
us = neutral_us + sign * us_per_rad * angle_rad, clipped to [min_us, max_us].
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np

SYNC = b"\xaa\x55"
N = 19
FLAG_RELAX = 0x01


def crc16_ccitt(data: bytes, crc: int = 0xFFFF) -> int:
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def encode_frame(seq: int, pulses_us, relax: bool = False) -> bytes:
    pulses = [int(p) for p in pulses_us]
    if len(pulses) != N or any(not 500 <= p <= 2500 for p in pulses):
        raise ValueError("19 pulses in [500, 2500] us required")
    body = struct.pack("<BB" + "H" * N, seq & 0xFF, FLAG_RELAX if relax else 0, *pulses)
    return SYNC + body + struct.pack("<H", crc16_ccitt(body))


def decode_frame(frame: bytes):
    if len(frame) != 2 + 2 + 2 * N + 2 or frame[:2] != SYNC:
        raise ValueError("bad length/sync")
    body, crc = frame[2:-2], struct.unpack("<H", frame[-2:])[0]
    if crc16_ccitt(body) != crc:
        raise ValueError("bad crc")
    seq, flags, *pulses = struct.unpack("<BB" + "H" * N, body)
    return seq, bool(flags & FLAG_RELAX), pulses


class SerialServoLink:
    """Drop-in ServoLink for robot_host.py."""

    def __init__(self, port, calibration_json: str, baud: int = 115200):
        """`port` is a COM-port name OR an already-open pyserial object (share it with SerialImu)."""
        table = json.loads(Path(calibration_json).read_text())
        if len(table) != N:
            raise ValueError("Calibration needs 19 joints")
        self.neutral = np.array([t["neutral_us"] for t in table], float)
        self.sign = np.array([t["sign"] for t in table], float)
        self.scale = np.array([t["us_per_rad"] for t in table], float)
        self.lo = np.array([t["min_us"] for t in table], float)
        self.hi = np.array([t["max_us"] for t in table], float)
        if not np.all(np.abs(self.sign) == 1) or np.any(self.scale <= 0) or np.any(self.lo >= self.hi):
            raise ValueError("Invalid calibration table")
        unverified = [t.get("sim_joint", i) for i, t in enumerate(table) if not t.get("sign_verified", False)]
        if unverified:
            raise ValueError("Refusing to drive servos: joint direction (sign) not verified for " + ", ".join(map(str, unverified)) +
                             ". Add a direction note in the console and re-run tools/csv_to_limits.py.")
        if isinstance(port, str):
            import serial  # pyserial
            port = serial.Serial(port, baud, timeout=0.001, write_timeout=0.02)
        self.link = port
        self.seq = 0

    def pulses(self, angles_rad):
        return np.clip(self.neutral + self.sign * self.scale * np.asarray(angles_rad, float), self.lo, self.hi)

    def send(self, angles_rad):
        self.seq = (self.seq + 1) & 0xFF
        self.link.write(encode_frame(self.seq, np.rint(self.pulses(angles_rad))))

    def hold(self):
        """Send nothing: after 100 ms the STM32 watchdog HOLDS every servo at its last pulse."""

    def relax(self):
        for _ in range(5):                      # repeat: a single dropped frame must not matter
            self.seq = (self.seq + 1) & 0xFF
            self.link.write(encode_frame(self.seq, np.full(N, 1500), relax=True))


def _selftest():
    pulses = list(range(1000, 1000 + N))
    frame = encode_frame(7, pulses)
    assert len(frame) == 44, len(frame)
    seq, relax, back = decode_frame(frame)
    assert (seq, relax, back) == (7, False, pulses)
    assert decode_frame(encode_frame(255, pulses, relax=True))[1] is True
    bad = bytearray(frame); bad[10] ^= 1
    try:
        decode_frame(bytes(bad)); raise SystemExit("CRC failed to catch corruption")
    except ValueError:
        pass
    assert crc16_ccitt(b"123456789") == 0x29B1   # CRC-16/CCITT-FALSE check value
    print("serial_servo_link selftest OK (44-byte frames, CRC check value 0x29B1)")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
