"""
# `trimkit` UR10 Example

Like many people who don't understand how motion planning works, I always wondered why it isn't just "scipy.optimize on a b-spline" for simple robot arm motion. I think the short answer was "because that's gonna be untenably slow, and there's no gradient."

But with the march of ecosystem progress I think that might be less true:

- `yourdfpy` loads a model nicely
- `trimesh.transformations.rotation_matrix` has supported `sympy.Symbol` angles for years! So you can multiply your 6+ rotation matrices symbolically, then use `sympy.lambdify` to make a single numpy-fast function that gives you forward kinematics.
- And now the last piece in the house of cards: `trimkit.CollisionScene` binds `coal` which fixed `fcl`'s convex distances and is fast(er than Mujoco, almost always, by a hair). This example runs a scipy optimizer on a spline, in around 30ms on a contrived UR10 scene finds a collision-free path.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import sympy
from robot_descriptions.loaders.yourdfpy import load_robot_description
from scipy.interpolate import BSpline
from scipy.optimize import minimize
from trimesh import Trimesh
from trimesh.creation import box
from trimesh.transformations import rotation_matrix
from yourdfpy import URDF

import trimkit
from trimkit import CollisionResult


@dataclass(frozen=True)
class Plan:
    """
    A joint-space path and how the optimizer found it.

    Attributes
    ----------
    path : BSpline
      Joint angles along the path, from 0 at the start to 1 at the goal.
    target : float
      The clearance the last round aimed for.
    checked : float
      The least clearance its dense check found, infinite when nothing
      came within the clearance asked for.
    rounds : int
      How many rounds of L-BFGS-B ran.
    evaluations : int
      How many times all the rounds evaluated the cost.
    seconds : float
      How long planning took.
    """

    path: BSpline
    target: float
    checked: float
    rounds: int
    evaluations: int
    seconds: float


def forward_kinematics(robot: URDF) -> Callable[..., np.ndarray]:
    """
    Forward kinematics as one numpy function, multiplied out symbolically
    once by walking the chain as the URDF writes it.

    Against yourdfpy's `update_cfg` and `get_transform` on fresh random
    poses of this UR10e's 13 links, after 0.3 s to build:

    | poses | allocating | into `out` | yourdfpy | faster |
    |------:|-----------:|-----------:|---------:|-------:|
    |     1 |    0.20 ms |    0.20 ms |  0.46 ms |     2x |
    |   101 |    0.22 ms |    0.22 ms |    42 ms |   190x |
    |  2001 |     3.1 ms |     1.3 ms |   840 ms |   640x |

    `faster` is yourdfpy over writing into `out`. One pose is nearly all
    the call's fixed cost: the win is in batches.

    Parameters
    ----------
    robot : URDF
      The robot, whose joints list each joint's parent before it.

    Returns
    -------
    forward : callable
      `forward(joint_angles, out=None)`, the `(n, l, 4, 4)` poses of every
      link in `robot.link_map` order for `(n, j)` joint angles in
      `robot.actuated_joint_names` order.

    Examples
    --------
    >>> robot = load_robot_description("ur10e_description")
    >>> forward = forward_kinematics(robot)
    >>> poses = forward(np.zeros((1, robot.num_actuated_joints)))
    >>> poses.shape
    (1, 13, 4, 4)
    >>> tool = list(robot.link_map).index("tool0")
    >>> bool(np.allclose(poses[0, tool], robot.get_transform("tool0")))
    True
    >>> out = np.zeros((101, 13, 4, 4))
    >>> forward(np.zeros((101, robot.num_actuated_joints)), out=out) is out
    True
    """
    links = list(robot.link_map)
    angles = sympy.symbols(f"q:{robot.num_actuated_joints}")
    turns = dict(zip(robot.actuated_joint_names, angles, strict=True))
    pose = {robot.base_link: sympy.eye(4)}
    for joint in robot.robot.joints:  # each joint's parent before it
        turn = rotation_matrix(turns[joint.name], joint.axis) if joint.name in turns else sympy.eye(4)
        pose[joint.child] = pose[joint.parent] * sympy.Matrix(joint.origin) * turn
    entries = list(sympy.Matrix.vstack(*(pose[link] for link in links)))
    # numpy hands back the entries that never change as scalars, so fill those in once
    varies = np.array([bool(entry.free_symbols) for entry in entries])
    constants = np.array([0.0 if v else float(e) for e, v in zip(entries, varies, strict=True)])
    kinematics = sympy.lambdify(
        angles, [e for e, v in zip(entries, varies, strict=True) if v], modules="numpy", cse=True
    )

    def forward(joint_angles: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
        """
        The pose of every link at each set of joint angles.

        Parameters
        ----------
        joint_angles : (n, j) float64
          Each pose's actuated joint angles, in radians.
        out : (n, l, 4, 4) float64, optional
          Where to write the poses, skipping an allocation: any array numpy
          can view as `(n, l * 16)`, or it raises.

        Returns
        -------
        poses : (n, l, 4, 4) float64
          Each link's transform into the base frame at each pose: `out`
          when it's passed.
        """
        if out is None:
            out = np.zeros((len(joint_angles), len(links), 4, 4))
        # a view of `out`, or numpy raises rather than copy
        values = out.reshape(len(joint_angles), -1, copy=False)
        values[:] = constants
        values[:, varies] = np.array(kinematics(*joint_angles.T)).T
        return out

    return forward


class PlannableScene:
    """
    A serial arm of revolute joints and its tool among fixed convex
    obstacles, built once.

    Parameters
    ----------
    robot : URDF
      The arm, loaded with its collision scene graph, its meshes measured
      as their convex hulls.
    world : list of Trimesh
      Convex obstacles in the base frame, which never move.
    tool : list of Trimesh
      Convex meshes in the frame of `tool_link`, such as a gripper.
    tool_link : str
      The link carrying `tool`.
    joint_gap : int
      Fewer joints than this apart, two of the arm's meshes are never
      measured: they touch by design.
    """

    def __init__(
        self, robot: URDF, world: list[Trimesh], tool: list[Trimesh], tool_link: str, joint_gap: int
    ) -> None:
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
        # the link carrying each mesh (obstacles ride the base, which never
        # moves), and each joint's child, which turns about the joint's axis
        # in its own frame, from its own origin
        joints = robot.actuated_joints
        links = [robot.base_link] * len(world) + owners + [tool_link] * len(tool)
        self.links = list(robot.link_map)
        self.mesh_link = np.array([self.links.index(link) for link in links])
        self.joint_link = np.array([self.links.index(joint.child) for joint in joints])
        self.joint_axes = np.array([joint.axis for joint in joints])

        self.forward = forward_kinematics(robot)
        self.limits = np.array([(joint.limit.lower, joint.limit.upper) for joint in joints])

        # which joints move each mesh, every joint above its link in the chain:
        # measure every obstacle against the arm, and the arm's meshes
        # `joint_gap` joints apart
        names = np.array(robot.actuated_joint_names)
        moved = {robot.base_link: np.zeros(len(names), dtype=bool)}
        for joint in robot.robot.joints:  # each joint's parent before it
            moved[joint.child] = moved[joint.parent] | (names == joint.name)
        self.moved = np.array([moved[link] for link in links])
        depth = self.moved.sum(axis=1)
        a, b = np.triu_indices(self.mesh_count, 1)
        keep = (b >= len(world)) & ((a < len(world)) | (abs(depth[a] - depth[b]) >= joint_gap))
        self.pairs = np.stack([a[keep], b[keep]], axis=1)

    def measure(self, poses: np.ndarray, within: float) -> CollisionResult:
        """
        The pairs closer than `within` at any waypoint, and where, from one
        `collides` call over every waypoint.

        Parameters
        ----------
        poses : (n, l, 4, 4) float64
          Every link's pose at each waypoint, from `forward`.
        within : float
          How close a pair must come to be found.

        Returns
        -------
        found : CollisionResult
          Each pair closer than `within` at a waypoint, with its distance,
          witness points and normal. Instance `w * m + i` is mesh `i` at
          waypoint `w`, for `m` meshes.
        """
        count = len(poses)
        matrices = poses[:, self.mesh_link].reshape(-1, 4, 4)
        index = np.tile(np.arange(self.mesh_count, dtype=np.int64), count)
        pairs = (self.pairs + self.mesh_count * np.arange(count, dtype=np.int64)[:, None, None]).reshape(
            -1, 2
        )
        return self.scene.collides(matrices, index, pairs=pairs, margin=within, return_distance=True)

    def plan(self, guess: np.ndarray, *, clearance: float, smoothing: float, waypoint_count: int) -> Plan:
        """
        Bend a clamped cubic spline, its ends fixed, until every pair is
        `clearance` apart.

        Each round optimizes at `waypoint_count` waypoints then checks twenty
        times as many, and when the check comes up short the next round aims
        twice the shortfall farther out, for at most five rounds.

        Parameters
        ----------
        guess : (c, j) float64
          The spline's control points to start from, the first at the start
          and the last at the goal, which never move.
        clearance : float
          How far apart every measured pair should stay.
        smoothing : float
          How much the spline's bending weighs against its shortfall.
        waypoint_count : int
          How many waypoints along the spline each evaluation measures.

        Returns
        -------
        plan : Plan
          The round whose dense check found the most clearance.
        """
        controls = np.array(guess, dtype=np.float64)
        joint_count = len(self.limits)
        knots = np.concatenate([[0, 0, 0], np.linspace(0, 1, len(controls) - 2), [1, 1, 1]])
        weights = BSpline.design_matrix(np.linspace(0, 1, waypoint_count), knots, 3).toarray()
        bend = np.diff(np.eye(len(controls)), 2, axis=0)
        bounds = np.tile(self.limits, (len(controls) - 2, 1))
        checks = np.linspace(0, 1, 20 * (waypoint_count - 1) + 1)
        poses = np.zeros((waypoint_count, len(self.links), 4, 4))
        clear = False  # whether the last evaluation had every waypoint clear of its target

        def cost(free: np.ndarray, target: float) -> tuple[float, np.ndarray]:
            # squared shortfall in clearance plus bending, and its gradient
            nonlocal clear
            controls[1:-1] = free.reshape(-1, joint_count)
            self.forward(weights @ controls, out=poses)
            found = self.measure(poses, target)
            waypoint, mesh = np.divmod(found.pairs, self.mesh_count)
            waypoint = waypoint[:, 0]
            shortfall = np.maximum(target - found.distances, 0.0)
            clear = not shortfall.any()
            # how fast each side's closest point moves along the normal as each
            # joint turns, `(axis x (point - origin)) . normal`, taken as
            # `(point - origin) . (normal x axis)`, where the joint moves it
            child = poses[:, self.joint_link]
            axes = (child[..., :3, :3] @ self.joint_axes[..., None])[..., 0]
            turn = np.cross(found.normals[:, None], axes[waypoint])
            along = np.einsum("sik,sjk->sij", found.points, turn)
            along -= np.einsum("sjk,sjk->sj", child[waypoint, :, :3, 3], turn)[:, None]
            along *= self.moved[mesh]
            slope = along[:, 1] - along[:, 0]
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
            checked = self.measure(self.forward(path(checks)), clearance).distances.min(initial=np.inf)
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
        f" {waypoint_count * len(scene.pairs):,} pairs each, aiming for {plan.target * 1e3:.1f} mm"
        f" over {rounds}\nthen twenty times as many waypoints checked {checked}"
    )
