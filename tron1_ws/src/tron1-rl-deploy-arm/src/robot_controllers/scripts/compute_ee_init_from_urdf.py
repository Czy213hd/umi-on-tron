#!/usr/bin/env python3

import argparse
import math
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


@dataclass(frozen=True)
class Joint:
    name: str
    joint_type: str
    parent: str
    child: str
    origin_xyz: np.ndarray  # (3,)
    origin_rpy: np.ndarray  # (3,) roll,pitch,yaw
    axis: np.ndarray  # (3,)


def _parse_vec(text: str | None, n: int) -> np.ndarray:
    if text is None or text.strip() == "":
        return np.zeros(n, dtype=float)
    parts = [p for p in text.replace(",", " ").split() if p]
    if len(parts) != n:
        raise ValueError(f"Expected {n} floats, got {len(parts)}: {text!r}")
    return np.array([float(p) for p in parts], dtype=float)


def rpy_to_rot(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr = math.cos(roll)
    sr = math.sin(roll)
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cy = math.cos(yaw)
    sy = math.sin(yaw)

    # R = Rz(yaw) * Ry(pitch) * Rx(roll)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=float,
    )


def rot_to_rpy(R: np.ndarray) -> np.ndarray:
    # Match ControllerBase.h:getRpyFromRotationMatrix()
    pitch = math.atan2(-R[2, 0], math.sqrt(R[2, 1] * R[2, 1] + R[2, 2] * R[2, 2]))
    yaw = math.atan2(R[1, 0], R[0, 0])
    roll = math.atan2(R[2, 1], R[2, 2])
    return np.array([roll, pitch, yaw], dtype=float)


def axis_angle_to_rot(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    n = float(np.linalg.norm(axis))
    if n < 1.0e-12:
        return np.eye(3, dtype=float)
    x, y, z = axis / n
    c = math.cos(angle)
    s = math.sin(angle)
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=float,
    )


def make_T(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[:3, :3] = rpy_to_rot(float(rpy[0]), float(rpy[1]), float(rpy[2]))
    T[:3, 3] = xyz
    return T


def inv_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    p = T[:3, 3]
    Ti = np.eye(4, dtype=float)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ p
    return Ti


def joint_T_parent_child(j: Joint, q: float) -> np.ndarray:
    T = make_T(j.origin_xyz, j.origin_rpy)
    jt = j.joint_type
    if jt in ("revolute", "continuous"):
        R = axis_angle_to_rot(j.axis, q)
        Tq = np.eye(4, dtype=float)
        Tq[:3, :3] = R
        return T @ Tq
    if jt == "prismatic":
        Tq = np.eye(4, dtype=float)
        axis = j.axis
        n = float(np.linalg.norm(axis))
        if n > 1.0e-12:
            axis = axis / n
        Tq[:3, 3] = axis * q
        return T @ Tq
    # fixed / planar / floating -> treat as fixed
    return T


def parse_urdf_joints(urdf_path: Path) -> dict[str, Joint]:
    root = ET.parse(urdf_path).getroot()
    joints: dict[str, Joint] = {}
    for je in root.findall("joint"):
        name = je.get("name")
        jt = je.get("type", "fixed")
        if not name:
            continue
        parent_el = je.find("parent")
        child_el = je.find("child")
        if parent_el is None or child_el is None:
            continue
        parent = parent_el.get("link")
        child = child_el.get("link")
        if not parent or not child:
            continue

        origin_el = je.find("origin")
        xyz = _parse_vec(origin_el.get("xyz") if origin_el is not None else None, 3)
        rpy = _parse_vec(origin_el.get("rpy") if origin_el is not None else None, 3)
        axis_el = je.find("axis")
        axis = _parse_vec(axis_el.get("xyz") if axis_el is not None else "1 0 0", 3)

        joints[name] = Joint(
            name=name,
            joint_type=jt,
            parent=parent,
            child=child,
            origin_xyz=xyz,
            origin_rpy=rpy,
            axis=axis,
        )
    return joints


def build_link_graph(joints: dict[str, Joint]) -> dict[str, list[tuple[str, str, int]]]:
    adj: dict[str, list[tuple[str, str, int]]] = {}
    for j in joints.values():
        adj.setdefault(j.parent, []).append((j.child, j.name, +1))
        adj.setdefault(j.child, []).append((j.parent, j.name, -1))
    return adj


def find_path(
    adj: dict[str, list[tuple[str, str, int]]], base_link: str, ee_link: str
) -> list[tuple[str, int]]:
    if base_link == ee_link:
        return []
    from collections import deque

    q = deque([base_link])
    prev: dict[str, tuple[str, str, int]] = {base_link: ("", "", 0)}
    while q:
        cur = q.popleft()
        for nxt, joint_name, direction in adj.get(cur, []):
            if nxt in prev:
                continue
            prev[nxt] = (cur, joint_name, direction)
            if nxt == ee_link:
                q.clear()
                break
            q.append(nxt)

    if ee_link not in prev:
        raise RuntimeError(f"No kinematic path found: {base_link} -> {ee_link}")

    # reconstruct as (joint_name, direction) from base -> ee
    path: list[tuple[str, int]] = []
    cur = ee_link
    while cur != base_link:
        p, joint_name, direction = prev[cur]
        path.append((joint_name, direction))
        cur = p
    path.reverse()
    return path


def load_params(params_path: Path | None) -> dict:
    if params_path is None:
        return {}
    with params_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_nested(d: dict, keys: list[str], default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Compute EE init pose (pos+rpy) from URDF + default joint angles in params.yaml."
    )
    ap.add_argument("--urdf", default="/home/phi5090ii/NYX/umi-on-tron-lab/tron1_ws/src/tron1-rl-deploy-arm/src/robot-description/pointfoot/SF_TRON1A_ARX5ARM/urdf/robot_with_arm.urdf", type=Path, help="Path to URDF file")
    ap.add_argument("--params", default="/home/phi5090ii/NYX/umi-on-tron-lab/tron1_ws/src/tron1-rl-deploy-arm/src/robot_controllers/config/pointfoot/SF_TRON1A_ARX5ARM/params.yaml", type=Path, help="Path to params.yaml (PointfootCfg)")
    ap.add_argument("--base-link", type=str, help="Base link name (default from params or 'base_Link')")
    ap.add_argument("--ee-link", type=str, help="EE link name (default from params or 'link6')")
    ap.add_argument(
        "--tip-offset-pos",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="Optional: link6->tip translation (meters), to also print TIP pose.",
    )
    ap.add_argument(
        "--tip-offset-rpy",
        type=float,
        nargs=3,
        metavar=("R", "P", "Y"),
        help="Optional: link6->tip rpy (rad), to also print TIP pose.",
    )
    ap.add_argument("--print-chain", action="store_true", help="Print joint chain used for FK.")
    args = ap.parse_args()

    params = load_params(args.params)
    base_link = args.base_link or get_nested(params, ["PointfootCfg", "ee", "base_link"], "base_Link")
    ee_link = args.ee_link or get_nested(params, ["PointfootCfg", "ee", "ee_link"], "link6")
    default_angles = get_nested(params, ["PointfootCfg", "init_state", "default_joint_angle"], {}) or {}
    joint_angles = {str(k): float(v) for k, v in default_angles.items()}

    joints = parse_urdf_joints(args.urdf)
    adj = build_link_graph(joints)
    path = find_path(adj, base_link, ee_link)

    if args.print_chain:
        print(f"FK chain {base_link} -> {ee_link}:")
        for joint_name, direction in path:
            j = joints[joint_name]
            arrow = "->" if direction == +1 else "<-"
            print(f"  {j.parent} {arrow}({joint_name}) {j.child}  q={joint_angles.get(joint_name, 0.0):.6f}")

    T = np.eye(4, dtype=float)
    for joint_name, direction in path:
        j = joints[joint_name]
        q = joint_angles.get(joint_name, 0.0)
        T_pc = joint_T_parent_child(j, q)
        T = T @ (T_pc if direction == +1 else inv_T(T_pc))

    pos = T[:3, 3]
    rpy = rot_to_rpy(T[:3, :3])

    print(f"base_link: {base_link}")
    print(f"ee_link: {ee_link}")
    print(f"ee_init_pos (m): [{pos[0]: .6f}, {pos[1]: .6f}, {pos[2]: .6f}]")
    print(f"ee_init_rpy (rad): [{rpy[0]: .6f}, {rpy[1]: .6f}, {rpy[2]: .6f}]")
    print()
    print("C++:")
    print(f"  ee_init_pos_ << {pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f};")
    print(f"  ee_init_rpy_ << {rpy[0]:.6f}, {rpy[1]:.6f}, {rpy[2]:.6f};")

    if args.tip_offset_pos is not None and args.tip_offset_rpy is not None:
        T_tip = make_T(np.array(args.tip_offset_pos, dtype=float), np.array(args.tip_offset_rpy, dtype=float))
        T_base_tip = T @ T_tip
        tip_pos = T_base_tip[:3, 3]
        tip_rpy = rot_to_rpy(T_base_tip[:3, :3])
        print()
        print("TIP (base -> tip):")
        print(f"tip_pos (m): [{tip_pos[0]: .6f}, {tip_pos[1]: .6f}, {tip_pos[2]: .6f}]")
        print(f"tip_rpy (rad): [{tip_rpy[0]: .6f}, {tip_rpy[1]: .6f}, {tip_rpy[2]: .6f}]")

    return 0


if __name__ == "__main__":
    sys.exit(main())

