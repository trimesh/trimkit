"""A UR10e with a box gripper reaches into a cubby for a block. The planner is one function, forward
kinematics then `distances` over a joint-space spline's waypoints, in `scipy.optimize.minimize`."""

import time

import numpy as np
from robot_descriptions.loaders.yourdfpy import load_robot_description as load
from scipy.interpolate import BSpline
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from trimesh.creation import box

import trimkit

CLEARANCE, SMOOTH, CONTROLS = 0.025, 1e-5, 6  # m between every pair, the bending's weight, spline size
GOAL = np.array([-0.326207, -1.521732, 2.080108, -0.558376, 1.244589, 0.0])  # fingers around the block
START = GOAL + 1.3 * np.eye(6)[0]  # swung off to one side
robot = load("ur10e_description", build_collision_scene_graph=True, load_collision_meshes=True)
joints, actuated = robot.robot.joints, [j.name for j in robot.actuated_joints]


def forward(q: np.ndarray):
    # for n configurations: every link's (n, 4, 4) pose, and each joint's world axis and origin
    frames = {robot.base_link: np.tile(np.eye(4), (len(q), 1, 1))}
    axes, origins = [], []
    for joint in joints:
        frame = frames[joint.parent] @ joint.origin
        if joint.name in actuated:
            axes.append(frame[:, :3, :3] @ joint.axis)
            origins.append(frame[:, :3, 3])
            turn = Rotation.from_rotvec(q[:, len(axes) - 1, None] * joint.axis).as_matrix()
            frame[:, :3, :3] = frame[:, :3, :3] @ turn
        frames[joint.child] = frame
    return frames, np.stack(axes, 1), np.stack(origins, 1)


# the arm as convex hulls and a three-box gripper, all where they are at rest
rest, graph = forward(np.zeros((1, 6)))[0], robot.collision_scene.graph
nodes, meshes = graph.nodes_geometry, robot.collision_scene.geometry
shapes = [meshes[graph[n][1]].convex_hull.apply_transform(graph.get(n)[0]) for n in nodes]
hand = [((-0.08, -0.03, 0), (0.08, 0.03, 0.04))]
hand += [((x, -0.02, 0.04), (x + 0.02, 0.02, 0.2)) for x in (-0.08, 0.06)]
shapes += [box(bounds=b).apply_transform(rest["tool0"][0]) for b in hand]
links = [graph.transforms.parents[n] for n in nodes] + ["tool0"] * 3
undo = np.stack([np.linalg.inv(rest[link][0]) for link in links])

# the world: a cubby on a stand, its walls thicker than any overlap, and the block inside it
world = [
    ((0.6, -0.3, 0), (1.1, 0.3, 0.3)),
    ((0.6, -0.3, 0.65), (1.1, 0.3, 0.75)),
    ((0.6, -0.3, 0.3), (1.1, -0.2, 0.65)),
    ((0.6, 0.2, 0.3), (1.1, 0.3, 0.65)),
    ((1, -0.2, 0.3), (1.1, 0.2, 0.65)),
    ((0.79, -0.03, 0.3), (0.85, 0.03, 0.44)),
]
shapes = [box(bounds=b) for b in world] + shapes
scene = trimkit.CollisionScene([(m.vertices, m.faces) for m in shapes], convex=True)

# joints moving each instance of a serial arm: bodies pair with the world, and with bodies 3 joints away
depth = {robot.base_link: 0}
for joint in joints:
    depth[joint.child] = depth[joint.parent] + (joint.name in actuated)
depths = np.array([-3] * len(world) + [depth[link] for link in links])
moved, template = np.arange(6) < depths[:, None], np.argwhere(np.subtract.outer(depths, depths) >= 3)
# a clamped cubic spline on 2001 waypoints, of which the optimizer looks at every 20th
knots = np.r_[[0] * 3, np.linspace(0, 1, CONTROLS - 2), [1] * 3]
dense = BSpline.design_matrix(np.linspace(0, 1, 2001), knots, 3).toarray()
bend = np.diff(np.eye(CONTROLS), 2, axis=0)


def measure(q: np.ndarray, **options):
    frames, axes, origins = forward(q)
    placed = (np.stack([frames[link] for link in links], 1) @ undo).reshape(-1, 4, 4)
    instances = np.r_[np.tile(np.eye(4), (len(world), 1, 1)), placed]
    index = np.r_[np.arange(len(world)), np.tile(np.arange(len(world), len(shapes)), len(q))]
    pairs = template + (template >= len(world)) * len(links) * np.arange(len(q))[:, None, None]
    return (*scene.distances(instances, index, pairs=pairs.reshape(-1, 2), **options), axes, origins)


def cost(x: np.ndarray):
    # each pair's squared shortfall, whose slope is the normal along each side's motion w x (p - o)
    controls = np.vstack([START, x.reshape(-1, 6), GOAL])
    distance, witness, normal, axes, origins = measure(dense[::20] @ controls, upper_bound=CLEARANCE)
    hit = np.flatnonzero(distance < CLEARANCE)
    short, w, row = CLEARANCE - distance[hit], hit // len(template), template[hit % len(template)]
    slope = np.zeros((len(hit), 6))
    for side, sign in ((1, 1), (0, -1)):
        motion = np.cross(axes[w], witness[hit, side][:, None] - origins[w])
        slope += sign * np.einsum("pjk,pk->pj", motion, normal[hit]) * moved[row[:, side]]
    gradient = SMOOTH * bend.T @ bend @ controls - dense[::20][w].T @ (short[:, None] * slope)
    return 0.5 * (short @ short + SMOOTH * np.sum((bend @ controls) ** 2)), gradient[1:-1].ravel()


limits = [(j.limit.lower, j.limit.upper) for j in robot.actuated_joints] * (CONTROLS - 2)
straight = np.linspace(START, GOAL, CONTROLS)
tic = time.perf_counter()
solved = minimize(cost, straight[1:-1].ravel(), jac=True, method="L-BFGS-B", bounds=limits)
print(f"planned in {(time.perf_counter() - tic) * 1e3:.0f} ms, {solved.nfev} evaluations of 101 waypoints")
path = np.vstack([START, solved.x.reshape(-1, 6), GOAL])
print(f"least clearance over 2001 waypoints: {measure(dense @ path)[0].min() * 1e3:.1f} mm")
