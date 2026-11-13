"""
Motion planning with `scipy.optimize.minimize`: a UR10e with a Robotiq 2F-85
reaches into a cubby for a block. The scene and its forward kinematics are
built once, and every evaluation of the path is one trimkit `distances` call
over all of its waypoints.
"""

import time
from dataclasses import dataclass

import numpy as np
import sympy
from robot_descriptions.loaders.yourdfpy import load_robot_description
from scipy.interpolate import BSpline
from scipy.optimize import minimize
from trimesh import Trimesh
from trimesh.creation import box
from trimesh.geometry import align_vectors
from trimesh.transformations import rotation_matrix
from yourdfpy import URDF

import trimkit


@dataclass(frozen=True)
class Plan:
    """A joint-space path from 0 at the start to 1 at the goal, and how the optimizer found it."""

    path: BSpline
    target: float  # the clearance the last round aimed for
    checked: float  # the least clearance its dense check found, infinite when clear
    rounds: int
    evaluations: int
    seconds: float


class PlannableScene:
    """A serial arm of revolute joints and its tool among fixed convex obstacles, built once."""

    def __init__(
        self, robot: URDF, world: list[Trimesh], tool: list[Trimesh], tool_link: str, joint_gap: int
    ) -> None:
        """
        `robot` is loaded with its collision scene graph, `tool` is in the
        frame of `tool_link`, and the arm's meshes fewer than `joint_gap`
        joints apart are never measured: they touch by design.
        """
        # the arm's collision meshes as convex hulls in their links' frames
        graph = robot.collision_scene.graph
        owners = [graph.transforms.parents[node] for node in graph.nodes_geometry]
        placed = [graph.get(node, owner) for node, owner in zip(graph.nodes_geometry, owners, strict=True)]
        hulls = [
            robot.collision_scene.geometry[name].convex_hull.copy().apply_transform(matrix)
            for matrix, name in placed
        ]
        meshes = [*world, *hulls, *tool]
        self.mesh_count = len(meshes)
        self.scene = trimkit.CollisionScene([(mesh.vertices, mesh.faces) for mesh in meshes], convex=True)
        # the link carrying each mesh: obstacles ride the base, which never moves
        links = [robot.base_link] * len(world) + owners + [tool_link] * len(tool)

        # forward kinematics once, symbolically, walking the chain as the URDF writes it
        angles = sympy.symbols(f"q:{robot.num_actuated_joints}")
        turns = dict(zip(robot.actuated_joint_names, angles, strict=True))
        pose = {robot.base_link: sympy.eye(4)}
        for joint in robot.robot.joints:  # each joint's parent before it
            turn = rotation_matrix(turns[joint.name], joint.axis) if joint.name in turns else sympy.eye(4)
            pose[joint.child] = pose[joint.parent] * sympy.Matrix(joint.origin) * turn
        # every mesh's instance matrix, then each joint's frame turned so its z is the joint's axis
        joint_frames = [
            pose[joint.child] * sympy.Matrix(align_vectors([0, 0, 1], joint.axis))
            for joint in robot.actuated_joints
        ]
        entries = list(sympy.Matrix.vstack(*(pose[link] for link in links), *joint_frames))
        # numpy hands back the entries that never change as scalars, so fill those in once
        self.varies = np.array([bool(entry.free_symbols) for entry in entries])
        self.constants = np.array([0.0 if v else float(e) for e, v in zip(entries, self.varies, strict=True)])
        varying = [entry for entry, v in zip(entries, self.varies, strict=True) if v]
        self.kinematics = sympy.lambdify(angles, varying, modules="numpy", cse=True)
        self.limits = np.array([(joint.limit.lower, joint.limit.upper) for joint in robot.actuated_joints])

        # which joints move each mesh, read off its pose: measure every obstacle
        # against the arm, and the arm's meshes `joint_gap` joints apart
        self.moved = np.array([[angle in pose[link].free_symbols for angle in angles] for link in links])
        depth = self.moved.sum(axis=1)
        a, b = np.triu_indices(self.mesh_count, 1)
        keep = (b >= len(world)) & ((a < len(world)) | (abs(depth[a] - depth[b]) >= joint_gap))
        self.pairs = np.stack([a[keep], b[keep]], axis=1)

    def forward(self, angles: np.ndarray) -> np.ndarray:
        """
        `(n, m + j, 4, 4)` poses for `(n, j)` joint angles: every mesh's
        instance matrix, then each joint's frame, its z the joint's axis.
        """
        values = np.empty((len(angles), len(self.constants)))
        values[:] = self.constants
        values[:, self.varies] = np.array(self.kinematics(*angles.T)).T
        return values.reshape(len(angles), -1, 4, 4)

    def measure(
        self, poses: np.ndarray, upper_bound: float = np.inf
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Every pair's distance, closest points and normal at every waypoint, in
        one `distances` call: instance `w * m + i` is mesh `i` at waypoint `w`.
        """
        count = len(poses)
        matrices = poses[:, : self.mesh_count].reshape(-1, 4, 4)
        index = np.tile(np.arange(self.mesh_count), count)
        pairs = (self.pairs + self.mesh_count * np.arange(count)[:, None, None]).reshape(-1, 2)
        return self.scene.distances(matrices, index, pairs=pairs, upper_bound=upper_bound)

    def plan(self, guess: np.ndarray, *, clearance: float, smoothing: float, waypoint_count: int) -> Plan:
        """
        Bend a clamped cubic spline through the `(c, j)` control points `guess`,
        its ends fixed, until every pair is `clearance` apart, with `smoothing`
        weighing its bending. Each round optimizes at `waypoint_count` waypoints
        then checks twenty times as many, and when the check comes up short the
        next round aims twice the shortfall farther out.
        """
        controls = np.array(guess, dtype=np.float64)
        joint_count = len(self.limits)
        knots = np.concatenate([[0, 0, 0], np.linspace(0, 1, len(controls) - 2), [1, 1, 1]])
        weights = BSpline.design_matrix(np.linspace(0, 1, waypoint_count), knots, 3).toarray()
        bend = np.diff(np.eye(len(controls)), 2, axis=0)
        bounds = np.tile(self.limits, (len(controls) - 2, 1))
        checks = np.linspace(0, 1, 20 * (waypoint_count - 1) + 1)
        clear = False  # whether the last evaluation had every waypoint clear of its target

        def cost(free: np.ndarray, target: float) -> tuple[float, np.ndarray]:
            # squared shortfall in clearance plus bending, and its gradient
            nonlocal clear
            controls[1:-1] = free.reshape(-1, joint_count)
            poses = self.forward(weights @ controls)
            distances, points, normals = self.measure(poses, target)
            short = np.nonzero(distances < target)[0]
            clear = len(short) == 0
            shortfall = target - distances[short]
            waypoint, pair = np.divmod(short, len(self.pairs))
            # how fast each side's closest point moves along the normal as each
            # joint turns: `axis x (point - origin)`
            joints = poses[waypoint, self.mesh_count :]
            axes, origins = joints[:, :, :3, 2], joints[:, :, :3, 3]
            slope = np.zeros((len(short), joint_count))
            for side, sign in ((0, -1.0), (1, 1.0)):
                velocity = np.cross(axes, points[short, None, side] - origins)
                along = (velocity * normals[short, None]).sum(axis=-1)
                slope += sign * along * self.moved[self.pairs[pair, side]]
            bent = bend @ controls
            gradient = smoothing * bend.T @ bent - weights[waypoint].T @ (shortfall[:, None] * slope)
            value = 0.5 * (shortfall @ shortfall + smoothing * np.sum(bent**2))
            return value, gradient[1:-1].ravel()

        def stop(intermediate_result: object) -> None:
            # a smooth path clear of everything is done: don't polish its bend
            if clear:
                raise StopIteration

        tic = time.perf_counter()
        # aim a little past `clearance`: the optimizer stops right at its
        # target, and between waypoints the path dips below it
        target, rounds, evaluations, best = 1.1 * clearance, 0, 0, (-np.inf, None, None)
        while rounds < 5:
            rounds += 1
            solved = minimize(
                cost,
                controls[1:-1].ravel(),
                args=(target,),
                jac=True,
                method="L-BFGS-B",
                bounds=bounds,
                callback=stop,
            )
            evaluations += solved.nfev
            controls[1:-1] = solved.x.reshape(-1, joint_count)
            path = BSpline(knots, controls.copy(), 3)
            # infinite when nothing at all comes within `clearance`
            checked = self.measure(self.forward(path(checks)), clearance)[0].min()
            best = max(best, (checked, target, path), key=lambda round_: round_[0])
            if checked >= clearance:
                break
            target += 2 * (clearance - checked)
        checked, target, path = best
        return Plan(path, target, checked, rounds, evaluations, time.perf_counter() - tic)


if __name__ == "__main__":
    robot = load_robot_description(
        "ur10e_description", build_collision_scene_graph=True, load_collision_meshes=True
    )
    # a cubby on a stand, its walls thicker than any overlap, and the block inside it
    cubby = [
        box(bounds=bounds)
        for bounds in [
            ((0.6, -0.3, 0), (1.1, 0.3, 0.29)),
            ((0.6, -0.3, 0.65), (1.1, 0.3, 0.75)),
            ((0.6, -0.3, 0.29), (1.1, -0.2, 0.65)),
            ((0.6, 0.2, 0.29), (1.1, 0.3, 0.65)),
            ((1, -0.2, 0.29), (1.1, 0.2, 0.65)),
            ((0.79, -0.01, 0.29), (0.85, 0.01, 0.44)),
        ]
    ]
    # the gripper wide open, mounted on the arm's tool flange
    hand = load_robot_description(
        "robotiq_2f85_description", build_collision_scene_graph=True, load_collision_meshes=True
    )
    gripper = [mesh.convex_hull for mesh in hand.collision_scene.dump()]
    scene = PlannableScene(robot, cubby, gripper, tool_link="tool0", joint_gap=3)

    # from swung off to one side to the fingers around the block, starting as
    # a straight joint-space line
    goal = np.array([-0.326207, -1.521732, 2.080108, -0.558376, 1.244589, 0.0])
    start = goal + np.array([1.3, 0, 0, 0, 0, 0])
    waypoint_count = 101
    plan = scene.plan(
        np.linspace(start, goal, 6), clearance=0.025, smoothing=1e-4, waypoint_count=waypoint_count
    )
    checked = "clear" if np.isinf(plan.checked) else f"short at {plan.checked * 1e3:.1f} mm"
    rounds = f"{plan.rounds} round" + ("s" if plan.rounds > 1 else "")
    print(
        f"planned in {plan.seconds * 1e3:.0f} ms: {plan.evaluations} L-BFGS-B evaluations of"
        f" {waypoint_count * len(scene.pairs):,} pair distances each, aiming for {plan.target * 1e3:.1f} mm"
        f" over {rounds}\nthen twenty times as many waypoints checked {checked}"
    )
