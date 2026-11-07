"""
A UR10e with a box gripper reaches into a cubby for a block. The planner is
one function, forward kinematics then `distances` over a joint-space spline's
waypoints, in `scipy.optimize.minimize`.
"""

import time
from dataclasses import dataclass

import numpy as np
from robot_descriptions.loaders.yourdfpy import load_robot_description
from scipy.interpolate import BSpline
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from trimesh import Trimesh
from trimesh.creation import box
from yourdfpy import URDF

import trimkit

# the least distance to keep between every pair of bodies, in meters
CLEARANCE = 0.025
# how much bending the spline costs against a shortfall in clearance
SMOOTH = 1e-5
# control points of the joint-space spline, its two ends included
CONTROLS = 6
# waypoints of the finished path, of which the optimizer looks at every `STRIDE`th
WAYPOINTS = 2001
STRIDE = 20
# the fingers around the block
GOAL = np.array([-0.326207, -1.521732, 2.080108, -0.558376, 1.244589, 0.0])
# swung off to one side
START = GOAL + 1.3 * np.eye(6)[0]
# a cubby on a stand, its walls thicker than any overlap, and the block inside it
WORLD = [
    ((0.6, -0.3, 0), (1.1, 0.3, 0.3)),
    ((0.6, -0.3, 0.65), (1.1, 0.3, 0.75)),
    ((0.6, -0.3, 0.3), (1.1, -0.2, 0.65)),
    ((0.6, 0.2, 0.3), (1.1, 0.3, 0.65)),
    ((1, -0.2, 0.3), (1.1, 0.2, 0.65)),
    ((0.79, -0.03, 0.3), (0.85, 0.03, 0.44)),
]
# a palm and two fingers, in the frame of the arm's tool flange
GRIPPER = [
    ((-0.08, -0.03, 0), (0.08, 0.03, 0.04)),
    ((-0.08, -0.02, 0.04), (-0.06, 0.02, 0.2)),
    ((0.06, -0.02, 0.04), (0.08, 0.02, 0.2)),
]


@dataclass(frozen=True)
class Problem:
    """
    What every evaluation of the path needs, computed once.

    Attributes
    ----------
    robot
      The arm's kinematics.
    scene
      The world's boxes then the arm's hulls and gripper boxes.
    links
      The link moving each of the arm's bodies.
    undo : (b, 4, 4) float64
      Each of the arm's bodies from where it is at rest back to its link's frame.
    pairs : (p, 2) int64
      The pairs of bodies to measure, indexing the scene's meshes.
    moved : (m, 6) bool
      Whether each joint moves each mesh of the scene.
    waypoints : (WAYPOINTS, CONTROLS) float64
      The spline's waypoints as weights of its control points.
    bend : (CONTROLS - 2, CONTROLS) float64
      The second difference of the control points.
    """

    robot: URDF
    scene: trimkit.CollisionScene
    links: list[str]
    undo: np.ndarray
    pairs: np.ndarray
    moved: np.ndarray
    waypoints: np.ndarray
    bend: np.ndarray


def forward(robot: URDF, joints: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """
    Forward kinematics for a batch of configurations.

    Parameters
    ----------
    robot
      The arm's kinematics.
    joints : (n, 6) float64
      The joint angles of each configuration.

    Returns
    -------
    frames
      Every link's (n, 4, 4) pose, by name.
    axes : (n, 6, 3) float64
      Each joint's axis in the world.
    origins : (n, 6, 3) float64
      A point on each joint's axis in the world.
    """
    actuated = [joint.name for joint in robot.actuated_joints]
    frames = {robot.base_link: np.tile(np.eye(4), (len(joints), 1, 1))}
    axes = []
    origins = []
    for joint in robot.robot.joints:
        frame = frames[joint.parent] @ joint.origin
        if joint.name in actuated:
            axes.append(frame[:, :3, :3] @ joint.axis)
            origins.append(frame[:, :3, 3])
            turn = Rotation.from_rotvec(joints[:, len(axes) - 1, None] * joint.axis).as_matrix()
            frame[:, :3, :3] = frame[:, :3, :3] @ turn
        frames[joint.child] = frame
    return frames, np.stack(axes, 1), np.stack(origins, 1)


def arm_bodies(robot: URDF) -> tuple[list[Trimesh], list[str]]:
    """
    The arm as convex hulls and a three-box gripper, all where they are at rest.

    Parameters
    ----------
    robot
      The arm's kinematics and collision meshes.

    Returns
    -------
    bodies
      Each body at rest.
    links
      The link moving each body.
    """
    graph = robot.collision_scene.graph
    meshes = robot.collision_scene.geometry
    rest = forward(robot, np.zeros((1, 6)))[0]["tool0"][0]
    bodies = []
    links = []
    for node in graph.nodes_geometry:
        transform, name = graph.get(node)
        bodies.append(meshes[name].convex_hull.apply_transform(transform))
        links.append(graph.transforms.parents[node])
    for bounds in GRIPPER:
        bodies.append(box(bounds=bounds).apply_transform(rest))
        links.append("tool0")
    return bodies, links


def make_problem(robot: URDF) -> Problem:
    """
    Build the scene, the pairs worth measuring and the spline.

    Parameters
    ----------
    robot
      The arm's kinematics and collision meshes.

    Returns
    -------
    problem
      What every evaluation of the path needs.
    """
    bodies, links = arm_bodies(robot)
    rest = forward(robot, np.zeros((1, 6)))[0]
    undo = np.stack([np.linalg.inv(rest[link][0]) for link in links])
    shapes = [box(bounds=bounds) for bounds in WORLD] + bodies
    scene = trimkit.CollisionScene([(shape.vertices, shape.faces) for shape in shapes], convex=True)

    # joints moving each mesh of a serial arm: bodies pair with the world, and
    # with bodies 3 joints away
    actuated = [joint.name for joint in robot.actuated_joints]
    link_depth = {robot.base_link: 0}
    for joint in robot.robot.joints:
        link_depth[joint.child] = link_depth[joint.parent] + (joint.name in actuated)
    depths = np.array([-3] * len(WORLD) + [link_depth[link] for link in links])
    pairs = np.argwhere(np.subtract.outer(depths, depths) >= 3)
    moved = np.arange(6) < depths[:, None]

    # a clamped cubic spline
    knots = np.r_[[0] * 3, np.linspace(0, 1, CONTROLS - 2), [1] * 3]
    waypoints = BSpline.design_matrix(np.linspace(0, 1, WAYPOINTS), knots, 3).toarray()
    bend = np.diff(np.eye(CONTROLS), 2, axis=0)
    return Problem(robot, scene, links, undo, pairs, moved, waypoints, bend)


def measure(
    problem: Problem, joints: np.ndarray, upper_bound: float = np.inf
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Every pair's distance in every configuration, one `distances` call.

    Parameters
    ----------
    problem
      The scene and the pairs to measure.
    joints : (n, 6) float64
      The joint angles of each configuration.
    upper_bound
      How far to measure pairs that are far apart.

    Returns
    -------
    distances : (n * p,) float64
      Each pair's distance, configuration by configuration.
    points : (n * p, 2, 3) float64
      The closest point on each side of each pair.
    normals : (n * p, 3) float64
      The direction between each pair.
    axes : (n, 6, 3) float64
      Each joint's axis in the world.
    origins : (n, 6, 3) float64
      A point on each joint's axis in the world.
    """
    frames, axes, origins = forward(problem.robot, joints)
    world_count = len(WORLD)
    arm_count = len(problem.links)
    arm_poses = np.stack([frames[link] for link in problem.links], 1) @ problem.undo
    instances = np.r_[np.tile(np.eye(4), (world_count, 1, 1)), arm_poses.reshape(-1, 4, 4)]
    world_index = np.arange(world_count)
    arm_index = np.tile(np.arange(world_count, world_count + arm_count), len(joints))
    index = np.r_[world_index, arm_index]
    # each configuration's arm instances follow the last's, and the world's stay put
    shift = arm_count * np.arange(len(joints))[:, None, None]
    pairs = problem.pairs + (problem.pairs >= world_count) * shift
    distances, points, normals = problem.scene.distances(
        instances, index, pairs=pairs.reshape(-1, 2), upper_bound=upper_bound
    )
    return distances, points, normals, axes, origins


def cost(free: np.ndarray, problem: Problem) -> tuple[float, np.ndarray]:
    """
    Each pair's squared shortfall in clearance plus the spline's bending, whose
    slope is the normal along each side's motion, `axis x (point - origin)`.

    Parameters
    ----------
    free : ((CONTROLS - 2) * 6,) float64
      The spline's control points between its two ends.
    problem
      The scene, the pairs and the spline.

    Returns
    -------
    value
      The cost.
    gradient : ((CONTROLS - 2) * 6,) float64
      The cost's slope along each of `free`.
    """
    controls = np.vstack([START, free.reshape(-1, 6), GOAL])
    waypoints = problem.waypoints[::STRIDE]
    joints = waypoints @ controls
    distances, points, normals, axes, origins = measure(problem, joints, upper_bound=CLEARANCE)

    pair_count = len(problem.pairs)
    short = np.flatnonzero(distances < CLEARANCE)
    shortfall = CLEARANCE - distances[short]
    waypoint = short // pair_count
    pair = problem.pairs[short % pair_count]
    slope = np.zeros((len(short), 6))
    for side, sign in ((1, 1), (0, -1)):
        motion = np.cross(axes[waypoint], points[short, side][:, None] - origins[waypoint])
        along = np.einsum("pjk,pk->pj", motion, normals[short])
        slope += sign * along * problem.moved[pair[:, side]]

    bent = problem.bend @ controls
    gradient = SMOOTH * problem.bend.T @ bent - waypoints[waypoint].T @ (shortfall[:, None] * slope)
    value = 0.5 * (shortfall @ shortfall + SMOOTH * np.sum(bent**2))
    return value, gradient[1:-1].ravel()


def plan(problem: Problem) -> tuple[np.ndarray, int]:
    """
    Bend a straight joint-space line from `START` to `GOAL` until it clears everything.

    Parameters
    ----------
    problem
      The scene, the pairs and the spline.

    Returns
    -------
    controls : (CONTROLS, 6) float64
      The spline's control points, its two ends included.
    evaluations
      How many times the optimizer evaluated the path.
    """
    limits = [(joint.limit.lower, joint.limit.upper) for joint in problem.robot.actuated_joints]
    straight = np.linspace(START, GOAL, CONTROLS)
    solved = minimize(
        cost,
        straight[1:-1].ravel(),
        args=(problem,),
        jac=True,
        method="L-BFGS-B",
        bounds=limits * (CONTROLS - 2),
    )
    return np.vstack([START, solved.x.reshape(-1, 6), GOAL]), solved.nfev


if __name__ == "__main__":
    robot = load_robot_description(
        "ur10e_description", build_collision_scene_graph=True, load_collision_meshes=True
    )
    problem = make_problem(robot)

    tic = time.perf_counter()
    controls, evaluations = plan(problem)
    milliseconds = (time.perf_counter() - tic) * 1e3
    coarse = len(problem.waypoints[::STRIDE])
    print(f"planned in {milliseconds:.0f} ms, {evaluations} evaluations of {coarse} waypoints")

    least = measure(problem, problem.waypoints @ controls)[0].min()
    print(f"least clearance over {WAYPOINTS} waypoints: {least * 1e3:.1f} mm")
