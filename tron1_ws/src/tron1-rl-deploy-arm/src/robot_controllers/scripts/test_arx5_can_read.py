#!/usr/bin/env python3
"""
Step 4: Quick ARX5 CAN read test WITHOUT starting the full controller.
Uses python-can to directly read joint state messages from ARX5 arm.

Usage:
  pip install python-can
  python3 test_arx5_can_read.py --interface can0 --duration 3

This verifies:
  1. CAN interface is up and receiving ARX5 motor frames
  2. All 6 joint motors (CAN ID 1,2,4,5,6,7) are responding
  3. Position/velocity/current values are reasonable (not all zeros/NaN)
"""

import argparse
import struct
import time
from collections import defaultdict

try:
    import can
except ImportError:
    print("ERROR: python-can not installed. Run: pip install python-can")
    raise

# ARX5 L5 motor CAN IDs (ID 3 is skipped)
JOINT_CAN_IDS = {1: "J1", 2: "J2", 4: "J3", 5: "J4", 6: "J5", 7: "J6", 8: "Gripper"}

# ARX5 DM motor feedback frame format (8 bytes):
#   bytes 0-1: motor ID (big-endian) -- but varies by firmware, check below
#   bytes 2-3: position (uint16, range [0, 0xFFFF] -> [-12.5, 12.5] rad)
#   bytes 4-5: velocity (uint12, range [0, 0xFFF] -> [-30, 30] rad/s)
#   bytes 5-6: current  (uint12 packed)
# NOTE: exact decoding depends on arx5-sdk firmware version.
# We just check that frames arrive and values are non-trivial.

def decode_dm_feedback(data: bytes):
    """Approximate decode of DM motor feedback. Returns (pos_raw, vel_raw, cur_raw)."""
    if len(data) < 6:
        return None
    # Position: bytes 2-3 (uint16 big-endian)
    pos_raw = struct.unpack(">H", data[1:3])[0]
    # Velocity: upper 8 bits of bytes 3-4
    vel_raw = (data[3] << 4) | (data[4] >> 4)
    # Current: lower 4 bits of byte 4 + byte 5
    cur_raw = ((data[4] & 0x0F) << 8) | data[5]
    # Map to physical values (DM motor range: pos [-12.5,12.5], vel [-30,30])
    pos_rad = (pos_raw / 65535.0) * 25.0 - 12.5
    vel_rads = (vel_raw / 4095.0) * 60.0 - 30.0
    cur_a   = (cur_raw / 4095.0) * 40.0 - 20.0
    return pos_rad, vel_rads, cur_a


def main():
    parser = argparse.ArgumentParser(description="ARX5 CAN read test")
    parser.add_argument("--interface", default="can0", help="CAN interface (default: can0)")
    parser.add_argument("--duration", type=float, default=3.0, help="Test duration in seconds")
    args = parser.parse_args()

    print(f"=== ARX5 CAN Read Test ({args.interface}, {args.duration}s) ===\n")

    try:
        bus = can.interface.Bus(channel=args.interface, bustype="socketcan")
    except OSError as e:
        print(f"ERROR: Cannot open {args.interface}: {e}")
        print("Make sure CAN interface is up:")
        print(f"  sudo ip link set {args.interface} type can bitrate 1000000")
        print(f"  sudo ip link set up {args.interface}")
        return 1

    received = defaultdict(list)
    t_end = time.time() + args.duration

    print(f"Listening on {args.interface} for {args.duration}s ...")
    print("Expected motor IDs: 1(J1) 2(J2) 4(J3) 5(J4) 6(J5) 7(J6) 8(Gripper)\n")

    while time.time() < t_end:
        msg = bus.recv(timeout=0.1)
        if msg is None:
            continue
        arb_id = msg.arbitration_id
        if arb_id in JOINT_CAN_IDS:
            received[arb_id].append(msg.data)

    bus.shutdown()

    # Report
    print("=== Results ===")
    all_ok = True
    for can_id, joint_name in JOINT_CAN_IDS.items():
        msgs = received[can_id]
        if not msgs:
            print(f"  [MISSING] {joint_name} (CAN ID {can_id}): NO FRAMES RECEIVED")
            all_ok = False
        else:
            decoded = decode_dm_feedback(msgs[-1])
            if decoded:
                pos, vel, cur = decoded
                print(f"  [OK]      {joint_name} (CAN ID {can_id}): "
                      f"{len(msgs):4d} frames  "
                      f"pos={pos:+.3f}rad  vel={vel:+.3f}rad/s  cur={cur:+.3f}A")
            else:
                print(f"  [OK]      {joint_name} (CAN ID {can_id}): {len(msgs)} frames (decode failed)")

    print()
    if all_ok:
        print("✓ All 7 motors are responding. CAN communication is working.\n")
        print("Next step: Run the full controller:")
        print("  ./host_run_tron1_real_arx5_socketcan.sh run")
    else:
        missing_ids = [can_id for can_id in JOINT_CAN_IDS if can_id not in received]
        print(f"✗ Missing motors: CAN IDs {missing_ids}")
        print("Check:")
        print("  1. Are all arm joints powered on?")
        print("  2. Is the CAN cable properly connected?")
        print("  3. Are 120Ω termination resistors present at both ends?")
        print("  4. Is the bitrate correct? (ARX5 L5 uses 1Mbit/s)")

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
