"""Contact evidence, action validation, failure modes and full physical episodes."""

import ast
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from manipulation_simulator import (
    ManipulationChunkExecutor,
    ManipulationFault,
    ManipulationSimulator,
)
from run_s3_benchmark import run, scripted_episode


@pytest.fixture
def sim():
    with ManipulationSimulator() as simulator:
        yield simulator


@pytest.fixture(scope="module")
def completed():
    with ManipulationSimulator() as simulator:
        result = scripted_episode(simulator)
        yield simulator, result


@pytest.mark.parametrize("command", [0, 0.25, 0.5, 0.75, 1])
def test_gripper_continuous_semantics_and_symmetry(sim, command):
    sim.command_gripper(command)
    assert sim.get_gripper_state()["measured_opening_m"] == pytest.approx(0.095)
    sim.hold(2)
    state = sim.get_gripper_state()
    assert state["commanded_opening_m"] == pytest.approx(command * 0.095)
    assert state["measured_opening_m"] == pytest.approx(command * 0.095, abs=2e-5)
    assert state["left_finger_position_m"] == pytest.approx(
        state["right_finger_position_m"], abs=1e-7
    )
    assert not state["holding"]
    assert not sim.compute_manipulation_metrics()["object_grasped"]


@pytest.mark.parametrize(
    "bad", [float("nan"), float("inf"), -float("inf"), -0.001, 1.001, "bad", None]
)
def test_invalid_gripper_rejects_without_mutation(sim, bad):
    before = sim.get_gripper_state()
    with pytest.raises(ValueError):
        sim.command_gripper(bad)
    assert sim.get_gripper_state() == before


def test_gripper_setpoint_speed_is_bounded_and_objects_are_dynamic(sim):
    start = sim.get_gripper_state()["measured_opening_m"]
    sim.command_gripper(0)
    sim.step()
    state = sim.get_gripper_state()
    assert state["setpoint_opening_m"] == pytest.approx(start - 0.06 * sim.timestep_s)
    assert state["measured_opening_m"] > 0.09
    sim.hold(0.1)
    assert sim.get_object_pose("blue_block").position_m[2] < 0.06
    assert sim.get_object_pose("black_block").position_m[2] < 0.055


def test_reset_restores_every_state_and_deterministic_clock(sim):
    initial = sim.data.qpos.copy()
    sim.command_gripper(0.4)
    sim.step(333)
    assert sim.simulation_time_s == 333 * 0.002
    sim.reset()
    assert np.array_equal(sim.data.qpos, initial)
    assert np.all(sim.data.qvel == 0)
    assert sim.simulation_time_s == 0
    assert sim.get_gripper_state()["state"] == "OPEN"
    assert sim.contact_timeline == []
    assert sim.compute_manipulation_metrics()["max_lift_height"] == 0
    assert sim.fault is None


def test_arm_execution_exact_and_ik_does_not_touch_free_state(sim):
    executor = ManipulationChunkExecutor(sim)
    pose = sim.get_tool_pose()
    before = sim.data.qpos.copy(), sim.data.qvel.copy(), sim.simulation_time_s
    executor.kinematics.reset(sim.get_joint_position())
    result = executor.ik.solve(
        type(pose)(pose.position_m + [0, 0, 0.01], pose.rotation_matrix)
    )
    assert result.converged
    assert np.array_equal(sim.data.qpos, before[0])
    assert np.array_equal(sim.data.qvel, before[1])
    assert sim.simulation_time_s == before[2]
    sim.command_joint_target(result.q_rad)
    sim.settle()
    assert np.array_equal(sim.get_joint_position(), result.q_rad)
    assert sim.contact_summary()["max_arm_projection_error_rad"] < 1e-8
    executor.close()


@pytest.mark.parametrize(
    "bad",
    [
        [0] * 6,
        [0] * 8,
        [0, 0, 0, 0.1, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, -1],
        [0, 0, 0, 0, 0, 0, float("nan")],
    ],
)
def test_invalid_actions_stop_chunk_without_gripper_side_effects(sim, bad):
    executor = ManipulationChunkExecutor(sim)
    start = sim.get_tool_pose().position_m
    result = executor.execute_chunk([bad, [*start, 0, 0, 0, 0]])
    assert not result.succeeded
    assert result.failed_index == 0
    assert result.records == ()
    assert sim.simulation_time_s == 0
    assert sim.get_gripper_state()["commanded_normalized_gripper"] == 1
    executor.close()


def test_valid_waypoint_then_invalid_stops_remaining_chunk(sim):
    executor = ManipulationChunkExecutor(sim)
    p = sim.get_tool_pose().position_m
    result = executor.execute_chunk(
        [
            [*(p + [0, 0, 0.005]), 0, 0, 0, 0.8],
            [*(p + [0.1, 0, 0]), 0, 0, 0, 0],
            [*p, 0, 0, 0, 0],
        ]
    )
    assert result.failed_index == 1
    assert len(result.records) == 1
    assert "exceeds" in result.reason
    assert sim.get_tool_pose().position_m == pytest.approx(p + [0, 0, 0.005], abs=1e-6)
    assert sim.get_gripper_state()["commanded_normalized_gripper"] == pytest.approx(0.8)
    executor.close()


def test_full_episode_has_actual_grasp_lift_transport_release_placement(completed):
    sim, result = completed
    assert result["success"], result["failure"]
    stages = {s["name"]: s for s in result["stages"]}
    assert all(s["pass"] for s in stages.values())
    candidate = stages["grasp"]["gripper"]
    assert candidate["state"] == "HOLDING_CANDIDATE"
    assert candidate["bilateral_blue_contact"]
    assert not candidate["holding"]  # On table contact alone is not proof of holding.
    assert (
        0.059 < candidate["measured_opening_m"] < 0.061
    )  # Object blocks commanded closure.
    assert candidate["commanded_opening_m"] == 0
    for name in ("lifted", "transported"):
        assert stages[name]["gripper"]["holding"]
        assert stages[name]["gripper"]["bilateral_blue_contact"]
    assert not stages["released"]["gripper"]["holding"]
    metrics = result["metrics"]
    assert metrics["max_lift_height"] > 0.14
    assert metrics["slip_distance"] < sim.scene_spec.physics.max_slip_m
    assert metrics["inside_target"] and metrics["released"] and metrics["settled"]
    assert metrics["black_block_displacement"] < 0.0001
    assert metrics["final_gripper_opening"] == pytest.approx(0.095, abs=1e-5)
    pairs = metrics["collision_contact_summary"]["pair_tick_counts"]
    assert all(
        "black" not in pair or pair == "black_block_collision / work_surface"
        for pair in pairs
    )
    assert pairs["blue_block_collision / left_finger_pad"] > 0
    assert pairs["blue_block_collision / right_finger_pad"] > 0
    assert pairs["blue_block_collision / red_box_bottom"] > 0


def test_episode_waypoints_remain_bounded_and_orientation_pinned(completed):
    sim, result = completed
    previous = np.array([-0.3, -0.15, 0.15])
    times = []
    for record in result["waypoints"]:
        action = np.array(record["requested_action"])
        assert action.shape == (7,)
        assert np.linalg.norm(action[:3] - previous) <= 0.02 + 1e-6
        assert np.all(action[3:6] == 0)
        assert 0 <= action[6] <= 1
        assert record["ik"]["status"] == "converged"
        assert np.linalg.norm(record["position_residual_m"]) < 1e-6
        previous = np.array(record["achieved_pose"]["position_m"])
        times.append(record["simulation_time_s"])
    assert all(b > a for a, b in zip(times, times[1:]))


def test_full_episode_reproducible_on_same_runtime(sim, completed):
    _, previous = completed
    assert scripted_episode(sim) == previous
    # Reusing the same instance after reset must clear physics warm-start state too.
    assert scripted_episode(sim) == previous


def _fixture_object_pose(sim, position, quaternion=(1, 0, 0, 0)):
    """Adversarial test fixture ONLY; not a runtime/benchmark manipulation API."""
    joint = sim.model.joint("blue_block_free").id
    address = sim.model.jnt_qposadr[joint]
    sim.data.qpos[address : address + 7] = [*position, *quaternion]
    sim.data.qvel[:] = 0
    mujoco.mj_forward(sim.model, sim.data)


@pytest.mark.parametrize(
    ("position", "quaternion", "expected"),
    [
        ((-0.30, 0.08, 0.065), (1, 0, 0, 0), True),
        (
            (-0.24, 0.08, 0.065),
            (1, 0, 0, 0),
            False,
        ),  # Center inside; extent straddles wall.
        ((-0.265, 0.08, 0.065), (0.9238795325, 0, 0, 0.3826834324), False),
        ((-0.30, 0.08, 0.14), (1, 0, 0, 0), False),  # Above rim.
        ((-0.30, 0.08, 0.04), (1, 0, 0, 0), False),  # Through floor.
    ],
)
def test_placement_checks_all_rotated_corners(sim, position, quaternion, expected):
    _fixture_object_pose(sim, position, quaternion)
    assert sim.inside_target() is expected
    assert not sim.compute_manipulation_metrics()["success"]


def test_inside_target_and_open_command_alone_do_not_mean_success(sim):
    _fixture_object_pose(sim, (-0.3, 0.08, 0.070))
    assert sim.inside_target()
    sim.command_gripper(1)
    metrics = sim.compute_manipulation_metrics()
    assert not metrics["settled"]
    assert not metrics["object_grasped"]
    assert not metrics["success"]


def test_frictionless_block_cannot_fake_grasp_success(sim):
    sim.model.geom_friction[:] = 0
    result = scripted_episode(sim)
    assert not result["success"]
    assert result["failure"]["category"] in {"grasp_fail", "dropped"}
    assert not result["metrics"]["object_grasped"]


def test_fault_latches_and_explicit_reset_clears(sim):
    # Deliberately create a floor through the tool in a test fixture.
    gid = sim.model.geom("work_surface").id
    original = sim.model.geom_pos[gid].copy()
    sim.model.geom_pos[gid, 2] = 0.19
    with pytest.raises(ManipulationFault, match="collision_abort"):
        sim.step()
    time = sim.simulation_time_s
    executor = ManipulationChunkExecutor(sim)
    result = executor.execute_chunk([[*sim.get_tool_pose().position_m, 0, 0, 0, 0]])
    assert not result.succeeded
    assert sim.simulation_time_s == time
    with pytest.raises(ManipulationFault):
        sim.command_gripper(1)
    sim.model.geom_pos[gid] = original
    sim.reset()
    assert sim.fault is None
    executor.close()


def test_no_hardware_provider_imports_and_no_runtime_free_object_pose_assignments():
    prohibited = {
        "socket",
        "serial",
        "cv2",
        "openai",
        "requests",
        "httpx",
        "pyrealsense2",
    }
    for filename in (
        "benchmark_scene.py",
        "manipulation_simulator.py",
        "run_s3_benchmark.py",
    ):
        source = Path(filename).read_text()
        tree = ast.parse(source)
        imports = {
            alias.name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        imports |= {
            node.module.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        }
        assert imports.isdisjoint(prohibited)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if "data.qpos[" in ast.unparse(target):
                        assert (
                            ast.unparse(target)
                            == "self.data.qpos[self._qpos_addresses]"
                        )


def test_evidence_is_strict_json_complete_and_never_overwritten(tmp_path):
    output = tmp_path / "evidence"
    result = run(output)
    assert result["success"]
    expected = {
        "validation.json",
        "benchmark_result.json",
        "environment.json",
        "final_object_states.json",
        "contact_summary.json",
        "contact_timeline.json",
        "trajectory.jsonl",
    }
    assert expected <= {p.name for p in output.iterdir()}
    for path in output.glob("*.json"):
        json.loads(
            path.read_text(),
            parse_constant=lambda c: pytest.fail(f"nonfinite JSON: {c}"),
        )
    validation = json.loads((output / "validation.json").read_text())
    assert validation["repeat_entire_episode_exact_equal"]
    assert all(value == 0 for value in validation["side_effects"].values())
    assert validation["render"]["requested"] is False
    with pytest.raises(FileExistsError):
        run(output)
