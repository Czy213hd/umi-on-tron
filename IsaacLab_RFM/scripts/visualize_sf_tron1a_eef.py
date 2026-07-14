"""Display the SF-TRON1A + ARX R5 arm and its ``eef_link`` coordinate frame.

Run from the ``IsaacLab_RFM`` directory:

    python scripts/visualize_sf_tron1a_eef.py

The URDF conversion is forced on every launch.  This is intentional: Isaac
Lab's lazy URDF conversion does not detect changes to referenced mesh files.

The Isaac Sim default timeline ends after 100 frames.  This viewer makes the
timeline loop, so it stays open for inspection instead of stopping after about
1.7 seconds.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Visualize the eef_link frame of the SF-TRON1A ARX R5 model.")
parser.add_argument(
    "--marker-scale",
    type=float,
    default=0.15,
    help="Length scale of the RGB coordinate axes in metres (default: 0.15).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
import omni.timeline
from isaaclab.assets import Articulation
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sim import SimulationContext

from ext_loco.assets.limx import LIMX_SF_TRON1A_ARM


def detach_stop_shutdown_callback(sim: SimulationContext) -> None:
    """Keep Isaac Lab from closing the standalone viewer during ``sim.reset()``.

    ``SimulationContext.reset()`` may issue a transient Timeline STOP before it
    starts physics.  In this Isaac Lab version, the standalone STOP handler
    treats that event as a request to close the entire app.  The viewer owns
    its lifetime instead, so detach that handler before the first reset.
    """
    stop_handle = sim._app_control_on_stop_handle
    if stop_handle is not None:
        stop_handle.unsubscribe()
        sim._app_control_on_stop_handle = None
        print("[INFO] Detached Isaac Lab's automatic STOP-to-shutdown handler.")


def configure_inspection_timeline() -> None:
    """Prevent Isaac Sim's short default timeline from stopping this viewer."""
    timeline = omni.timeline.get_timeline_interface()
    timeline.set_end_time(1.0e6)
    timeline.set_looping(True)
    # Timeline edits are queued until commit(), so make them live immediately.
    timeline.commit()
    print(
        "[INFO] Timeline configured: "
        f"end={timeline.get_end_time():g}s, looping={timeline.is_looping()}."
    )


def get_eef_definition() -> tuple[str, str, str, str, str]:
    """Read the frame definition from the same URDF that will be imported."""
    urdf_path = Path(LIMX_SF_TRON1A_ARM.spawn.asset_path)
    root = ET.parse(urdf_path).getroot()
    eef_joint = next(
        (
            joint
            for joint in root.findall("joint")
            if (child := joint.find("child")) is not None and child.get("link") == "eef_link"
        ),
        None,
    )
    if eef_joint is None:
        raise RuntimeError(f"No URDF joint defines the 'eef_link' frame: {urdf_path}")

    parent = eef_joint.find("parent")
    origin = eef_joint.find("origin")
    if parent is None or origin is None:
        raise RuntimeError(f"Incomplete eef_link joint definition in: {urdf_path}")

    return (
        str(urdf_path),
        eef_joint.get("name", "<unnamed>"),
        eef_joint.get("type", "<unspecified>"),
        parent.get("link", "<missing>"),
        f"xyz={origin.get('xyz', '0 0 0')}, rpy={origin.get('rpy', '0 0 0')}",
    )


def design_scene() -> tuple[Articulation, VisualizationMarkers]:
    """Spawn one stationary robot and a marker that follows its EEF link."""
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/GroundPlane", ground_cfg)

    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    robot_cfg = LIMX_SF_TRON1A_ARM.replace(prim_path="/World/Robot")
    # Mesh-only changes do not invalidate Isaac Lab's lazy URDF/USD cache.
    robot_cfg.spawn.force_usd_conversion = True
    # This viewer is for inspecting kinematics, not for testing balance control.
    robot_cfg.spawn.rigid_props.disable_gravity = True
    robot = Articulation(cfg=robot_cfg)

    marker_cfg = FRAME_MARKER_CFG.copy()
    marker_cfg.prim_path = "/World/Visuals/eef_link"
    marker_cfg.markers["frame"].scale = (args_cli.marker_scale,) * 3
    eef_marker = VisualizationMarkers(marker_cfg)

    return robot, eef_marker


def main():
    urdf_path, joint_name, joint_type, parent_link, origin = get_eef_definition()
    print(f"[INFO] URDF source: {urdf_path}")
    print(
        f"[INFO] EEF definition: {joint_type} joint '{joint_name}', "
        f"{parent_link} -> eef_link, {origin}"
    )

    sim = SimulationContext(sim_utils.SimulationCfg(device=args_cli.device))
    detach_stop_shutdown_callback(sim)
    sim.set_camera_view(eye=[2.2, -2.2, 1.7], target=[0.0, 0.0, 1.1])

    robot, eef_marker = design_scene()
    sim.reset()
    # Configure after reset because reset may restore the stage's default
    # 100-frame Timeline range.
    configure_inspection_timeline()

    eef_ids, eef_names = robot.find_bodies("eef_link")
    if eef_names != ["eef_link"]:
        raise RuntimeError(f"Expected exactly one 'eef_link', found: {eef_names}")
    eef_id = eef_ids[0]

    # Write the configured default pose explicitly so the displayed frame is
    # the URDF/model's nominal configuration on every reset.
    root_state = robot.data.default_root_state.clone()
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
    robot.reset()
    sim.play()

    print(f"[INFO] Visualizing imported '{eef_names[0]}' (body index {eef_id}).")
    print("[INFO] RGB axes are the eef_link frame defined above, not an EE command/target frame.")
    print("[INFO] Close Isaac Sim to exit.")
    print(
        "[INFO] App state before loop: "
        f"kit_running={simulation_app.app.is_running()}, "
        f"stage_ready={simulation_app.context.get_stage() is not None}, "
        f"timeline_playing={not sim.is_stopped()}."
    )

    sim_dt = sim.get_physics_dt()
    step_count = 0
    while simulation_app.is_running():
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim_dt)

        # body_link_pose_w is the URDF link frame, not the link COM frame.
        eef_pose_w = robot.data.body_link_pose_w[:, eef_id]
        eef_marker.visualize(
            translations=eef_pose_w[:, :3],
            orientations=eef_pose_w[:, 3:7],  # Isaac Lab uses wxyz quaternions.
        )

        if step_count % 300 == 0:
            position = eef_pose_w[0, :3].detach().cpu().tolist()
            quaternion_wxyz = eef_pose_w[0, 3:7].detach().cpu().tolist()
            print(f"[INFO] eef_link world pose: pos={position}, quat_wxyz={quaternion_wxyz}")
        step_count += 1


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
