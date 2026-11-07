import base64
import time

import numpy as np
import pytest
from helpers import areas2d

import trimkit


def shoelace(ring: np.ndarray) -> float:
    x, y = ring.T
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def star(n: int, radius: float, center=(0.0, 0.0), seed: int = 0) -> np.ndarray:
    # a random star-shaped (so simple) counter-clockwise ring
    rng = np.random.default_rng(seed)
    angle = np.sort(rng.uniform(0, 2 * np.pi, n))
    r = radius * rng.uniform(0.6, 1.0, n)
    return np.column_stack([r * np.cos(angle), r * np.sin(angle)]) + center


@pytest.mark.parametrize("seed", range(20))
def test_area_and_winding(seed):
    shell = star(200, 10.0, seed=seed)
    # holes are small stars well inside the shell's minimum radius
    holes = [np.ascontiguousarray(star(30, 1.0, center=c, seed=seed + 100)[::-1]) for c in ((-3, 0), (3, 0))]
    # close the shell with a duplicate vertex like shapely does
    exterior = np.vstack([shell, shell[:1]])

    vertices, faces = trimkit.triangulate(exterior, holes)
    # the returned vertices are exactly the rings stacked in order
    assert np.array_equal(vertices, np.vstack([exterior, *holes]))
    areas = areas2d(vertices, faces)
    truth = shoelace(shell) + sum(shoelace(h) for h in holes)

    # every triangle is wound the same way, and together they tile exactly
    # the polygon: any overlap, gap or flipped triangle breaks this
    assert np.all(areas > 0)
    assert abs(areas.sum() - truth) < 1e-10 * truth
    assert faces.dtype == np.int64
    assert faces.min() >= 0
    assert faces.max() < len(vertices)


def test_inputs_in_place_or_refused():
    shell = star(50, 1.0)
    hole = np.ascontiguousarray(star(10, 0.2)[::-1])
    _, expected = trimkit.triangulate(shell, [hole])
    # read-only arrays are used in place, and interiors may be any sequence
    frozen = shell.copy()
    frozen.flags.writeable = False
    for interiors in ([hole], (hole,)):
        assert np.array_equal(trimkit.triangulate(frozen, interiors)[1], expected)
    # a ring of another dtype, or a list, is refused naming which one (the
    # exterior is covered with every other array in test_invariants)
    for exterior, interiors, name in [
        (shell, [hole.astype(np.float32)], r"interiors\[0\]"),
        (shell, [hole, hole.tolist()], r"interiors\[1\]"),
    ]:
        with pytest.raises(TypeError, match=name):
            trimkit.triangulate(exterior, interiors)


def test_no_holes_and_empty_rings():
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    vertices, faces = trimkit.triangulate(square)
    assert np.array_equal(vertices, square) and len(faces) == 2
    # an empty hole adds nothing, and an empty exterior has no triangles
    assert np.array_equal(trimkit.triangulate(square, [np.zeros((0, 2))])[1], faces)
    vertices, faces = trimkit.triangulate(np.zeros((0, 2)))
    assert vertices.shape == (0, 2) and faces.shape == (0, 3)


def test_many_holes():
    # a grid of square holes: every hole cut, and the area exact
    exterior = np.array([[0, 0], [41, 0], [41, 41], [0, 41]], dtype=float)
    holes = [
        np.array([[x, y], [x, y + 1], [x + 1, y + 1], [x + 1, y]], dtype=float)
        for x in range(1, 41, 2)
        for y in range(1, 41, 2)
    ]
    vertices, faces = trimkit.triangulate(exterior, holes)
    assert len(vertices) == 4 + 4 * len(holes)
    assert abs(areas2d(vertices, faces).sum() - (41 * 41 - len(holes))) < 1e-9


def test_bad_rings():
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    with pytest.raises(TypeError):
        trimkit.triangulate(square[:, :1])
    with pytest.raises(TypeError, match=r"interiors\[1\]"):
        trimkit.triangulate(square, [square * 0.5, "not a ring"])
    with pytest.raises(ValueError, match="exterior"):
        trimkit.triangulate(np.full((4, 2), np.nan))
    with pytest.raises(ValueError, match=r"interiors\[0\]"):
        trimkit.triangulate(square, [np.full((4, 2), np.inf)])


def _degenerate(kind: str, n: int) -> np.ndarray:
    angle = np.linspace(0, 2 * np.pi, n, endpoint=False)
    if kind == "square":
        # every side densely sampled, so almost every vertex is collinear
        s = np.linspace(0, 1, n // 4, endpoint=False)
        z, o = np.zeros_like(s), np.ones_like(s)
        return np.vstack([np.column_stack(c) for c in ((s, z), (o, s), (1 - s, o), (z, 1 - s))])
    radius = 1.0 + 0.1 * (np.arange(n) % 2) if kind == "gear" else np.ones(n)
    return np.column_stack([radius * np.cos(angle), radius * np.sin(angle)])


@pytest.mark.parametrize("kind", ["circle", "gear", "square"])
def test_degenerate_rings(kind):
    # exactly regular, cocircular and collinear rings are what trimesh makes
    # from arcs and sampled edges: they must tile exactly, use every vertex so
    # extrusions stay watertight, and not go quadratic (a 40k circle took 0.7 s
    # before vendor/patches/earcut-convex-ring.patch and takes milliseconds now)
    vertices, faces = trimkit.triangulate(_degenerate(kind, 4000))
    areas = areas2d(vertices, faces)
    assert np.all(areas >= 0)
    assert abs(areas.sum() - shoelace(vertices)) < 1e-10 * shoelace(vertices)
    assert len(np.unique(faces)) == len(vertices)


@pytest.mark.timing
def test_regular_rings_are_not_quadratic():
    big = _degenerate("circle", 100_000)
    tic = time.perf_counter()
    trimkit.triangulate(big)
    # 0.020 s release, 0.055 s under ASan, 5.1 s without
    # earcut-convex-ring.patch (measured 2026-11-01, i7-12700K)
    assert time.perf_counter() - tic < 1.0


@pytest.mark.timing
def test_many_holes_are_not_quadratic():
    # bridging holes walked every hole spliced in before (7.7 s for these
    # 10,000; vendor/patches/earcut-hole-bridge-blocks.patch makes it 0.3 s),
    # and the triangles still tile exactly the polygon minus its holes
    side = 100
    outer = np.array([[-1.0, -1.0], [side + 1.0, -1.0], [side + 1.0, side + 1.0], [-1.0, side + 1.0]])
    holes = [
        np.ascontiguousarray(
            np.array([[i + 0.2, j + 0.2], [i + 0.2, j + 0.8], [i + 0.8, j + 0.8], [i + 0.8, j + 0.2]])
        )
        for i in range(side)
        for j in range(side)
    ]
    tic = time.perf_counter()
    vertices, faces = trimkit.triangulate(outer, holes)
    # 0.34 s release (what CI runners run), 1.3 s under ASan, 7.7 s without
    # earcut-hole-bridge-blocks.patch (measured 2026-11-01, i7-12700K): 4 s
    # is ~12x release and still well under the regression
    assert time.perf_counter() - tic < 4.0
    expected = (side + 2.0) ** 2 - len(holes) * 0.36
    assert np.isclose(np.abs(areas2d(vertices, faces)).sum(), expected, rtol=1e-12)


# found by fuzz/earcut.cpp: earcut 3.2.4 itself (pristine, without our patches)
# leaves out or overlaps area on these valid polygons with holes. Each is
# (ring sizes, base64 little-endian float64 x, y pairs), exact to the bit
UPSTREAM_HOLE_BUGS = {
    "holes 1e-4 of the ring": (
        [5, 4, 3, 4],
        (
            "oWBfDwAA4D8AAAAAAAAAABRuUdE4GOA/t7ZKjgdICT9W5NvXpv3fv8W/dOZtIYk/Z3oNi2N3z782Z/xE7krdv3CTnVysKt4/"
            "SGbtzrRdxb9Xx171FwRkv3Q42h51MHC/a3IhcJ1vZL+jHz7lmdBvv3psuYvO1ma/xUPNQXtJcL+FLgy4/JljvzeMPKJwXnC/"
            "whTMA8PqRT9Lj4yifihwv+dkAG7H2z0/vP23mcf1bb8wkBOEhf4DP4AZcirrTXC/QPMxTNycXj+Q0qTfazBwvyt4uUFMgV4/"
            "xWylPhq9br9OudbjTixYP4AZcirrTXC/tt1diQ6yXj95csQVWk1wvw=="
        ),
    ),
    "15 holes 2% of the ring": (
        [39, 8, 5, 7, 12, 8, 5, 7, 12, 12, 8, 12, 8, 5, 7, 8],
        (
            "igkAYPsP5D8AAAAAAAAAAL4a4MjbH+M/jrJQYMlYWT83bYOz0afkP9Ij5PCbVWg/HWEyUEQr4z/2U7CeOCxxP5gyHikT9O8/"
            "9VrpQXa1mD94EDxNjf3fPy0Qc+o1CIk/FnPFfHn93z9TTuQHtmyJP3yGaB4O9t8/HBafJIA4mT9rknI8h97vP+c9058VFLY/"
            "eRW2ULq37z9NYeiByEu/P0X/qD0MKO8/ZUUHTGwxzT9OkPxkQ0TdP8aj4zknr8s/EzBivVSgxj+TLff89+7dP/nQH6ss5bU/"
            "1j5JmEKH3z9lAlPIQQuYP6AS8UX29t8/MRBz6jUIib94EDxNjf3fP/6FkIa3vba/T0XDk6x93z/Q4gzJee/Iv/XrKmcNY+8/"
            "Gl8he/kZz7/XMJL7fgrvPwKf1S4GlNK/2y3GVkGf7j839hVdkY/Vv82Y/E4UIe4/TF6XHqnW1r8LtUyYpeTtP8l6RMBBWO2/"
            "lAuNMHuG2T/t1WSs8zjvv2hBfmqCC8w/O0CAGoEB4L96CAFeRNtSv5LBdGJu/++/iGVaJDkQgr9uQHldIdXfv44rE2h8da+/"
            "GTS02cgK7b//aMqShBLYvyOUJp8wkdq/3q7sx37h0b/0fiU0q93gvyYSNJHGQde/SqgfKCSky785RhQR1yXsv8o/i0QU4rU/"
            "qUtx+mSH37+iBr7/MHXJP9UDh8ICXN2/X5QNtzmz3T+pegU7y9PHv3M3v6p/VOg/2KqbfdUck78Iwr2pvV/rPx2wU+WLu42/"
            "8WcDsxz/3z9CqHqYqRmAvwBeqKd5POA/0yHWejjhL7/yjwVlAADgPwI17/582SK/cIib6q+7oL/CgBJ3xL/Sv4P7sBADrqu/"
            "XaxDy9Lh0b9ANrqKjDiuv++ognm+1dG/uapavNUis7896qQ14XfTvxMZCnUrqbC/VXMwAYxp1L/WC4rfroWgv2RdwMzFzNK/"
            "RrraplOFoL/vLrAbwMjSv44lW3pBhaC/sIK56i3G0r8I3Aa7DejJPw/QKNb4xdK/BLtYmElMxz8+wwA32MLRv2zuj/pwBcQ/"
            "tRwfLDKj07/z0te07GDGPxisJ+q3Q9S/vMattS5nyD9qcCwyydHSv/y/Maa5ZcI/ro3nxbX6zb9xGWfm1QjCP4PjLVM2hcy/"
            "RlhyIclQuj/KjC4LDEnOv4MAYYl92rw/cPMzx5w00L9uqc/lOuXAPzqOQIRZFc6/dy2iOU/lwD8lfBjlehHOv5s0bqZe5cA/"
            "hQ+lU8QLzr+3MXdWqGjIP1AhLIRyCM6/IGmc/7bayT9vrrDsogXOvxjTmXwQnMc/dosS62OAzL8CBLH5G87HP2Zm9ilaIMy/"
            "U0VHI2Osxj92Qvlc1vHKv3rPkQRre8Y/i9JoBRopzL+OFToNZBLGP5/8fOG/28+/qcpi7p9nyD+D96uNWRXOvx5HDyADMMk/"
            "08vyLBwVzr+83VKaFHnJP5s/fARpFc6/xsXQ+MRnyD8EdwseyQnOvzl8mbVwbcg/lMXNEpoJzr+L0ahaNevPPxiAdCOOCc6/"
            "f2IwdpU8yz/fO0u6gCXMv75KN+DnsMo/A/iBZrp7zr8DknKrnb7OP4qht8q9xs+/eumJ9wHqzz86jkCEWRXOv0aDLskZ6s8/"
            "c0EfzpoQzr9IbuTtH+rPP29FQjm3Ds6/p3QouCXqzz+FD6VTxAvOvyAEBQOrqMq/EmCXmiqHxr8kJbMlb0TNv3BGR1zpgMS/"
            "3vi94aNF0L9f+YNGnUHIvzUNNAnML86/IxiVwqiCyb9sGV4IiinMv8mgnlLLnsa/FxDti38V0b9Em1po3eu9v1xj0mvxQ9G/"
            "7kbngt4Au7+au0tWmyjUv2aQKqudxr6/s+6IIWMB07/8p2KQTuHAv/YGj8w6xdG/CGG4EsQivr+bBA9icJDRvz/22Gf5Hb6/"
            "ljGd6fqv0b/urFYnrA6+v/gFrgFl6s8/80xBdi/8vb8c9wXp8SrQP/j4qBho+L2/SX1BQskUzz+jRIQNMDW4v4Lvbo3O/c0/"
            "/yRd56VIur87zR4YKI/NP+/Zck8OaLq/8U32BzV7zD+IGTgjqRHAvz5Kl3AZZM0/PRE5fVCZwb8v8Qp5VQ/QPyQ4Sgz+GcG/"
            "XGxWdnjezz9mnMUgP9y+vwWCcdAX6s8/uZ0a8qoYvr+0w4wNJerPPxjH1b+6D76/5FoNAifqzz/vRUQYBAq+v1rjeYnCx7I/"
            "BoXshfS5AT8rUsTb36u1Py9JjR62WR8/Gya/1ZIusT8jSh+Go5KIP/GH7c+pkrE/Hpvhlz+Tjj8kFTRGcJ6uPxLt2TO+vZg/"
            "vj1ey4/arT/K2LrgQQeOPxBW/+1zNqw/cMiH4Bsjjb8+FVG5scWyPwnW7m7Ulje/KA6qHHhWtD/KdZf8Ehw3v2Q7MRGb6LQ/"
            "rAV/D8K1N794Cy3O+8WyP5ITdntLfd2+Xni+R1PRsj+ytviKst63vn+R7khu5sA/AAAAAAAAAADmROzInG+4P4tDlJLWQI4/"
            "YxX6nEFYtz+q+l3DEItsv+2jcDOtc78/JxcydPrSi79uqc/lOuXAP5FEHJjBlje/OkN0t1LlwD9abAWrqjIsvzwuKtxY5cA/"
            "nlwVN1ekJL+bNG6mXuXAPzdpe4SBsRG/tzF3VqhoyD8GheyF9LkBPyBpnP+22sk/L0mNHrZZHz8Y05l8EJzHPyNKH4ajkog/"
            "AgSx+RvOxz8em+GXP5OOP1NFRyNjrMY/Eu3ZM769mD96z5EEa3vGP8rYuuBBB44/jhU6DWQSxj9wyIfgGyONv6nKYu6fZ8g/"
            "CdbubtSWN78eRw8gAzDJP8p1l/wSHDe/Kl5UmhR5yT+3HIsPwrU3v8bF0PjEZ8g/khN2e0t93b5boZm1cG3IP1eJ+oqy3re+"
            "uOR0A6uoyr8YgHQjjgmuP/Np5y8H6sy/4ua3vJBktT+pSXQPORnQv7WzEhNxN7Q/fP+KIOFo0L9Ch+GuZF2mP71tvq/UNM6/"
            "cpoDfJaJoD+rqqRqp6zKv0zP3W4Otqw/Idvdh08pzL/xWJe+/6qtP1DgkXJi38u/9BXkDtT5rT8a5Cd6RybDvxiAdCOOCa4/"
            "HgXWnAvCxb9Qc1qOSRGzP7bRnjrkCMm/5hrCc8Mfpz8v7VaAaK3Gv9OffYOVG6I/ZvmAfyanxL89fVdDC6utPygA/Y6bqMq/"
            "7GSO3j4nvj+zpsdOfwXLv6HcAOIeicA/lyvdkWln0L/Kb76bfky9P2C9NLpigM6/N7AjJn9Quj/l7UAQEgjMvyifMDRY8L0/"
            "MelAO32ey7/xCRDfIvW9PyVDXUqS3cu/QlOSH3AEvj8+HyjHg4mgvxiAdCOOCb4/ZTqaxyQGq78g39uZz//AP8aTsOONVrO/"
            "+84jYhuewT9ZIoprHRyyvxYCoWb057s/oqxRIrlprr+jlfF0kEm6P2rlGrd9iaa/qv5ls0zavT9n6UX9Noemv8oQYGa8Ar4/"
            "xYGW/BlLpr+r27hMawa+Pw=="
        ),
    ),
}


@pytest.mark.parametrize("name", UPSTREAM_HOLE_BUGS)
@pytest.mark.xfail(strict=True, reason="earcut 3.2.4 mis-tiles some polygons with holes (upstream)")
def test_hole_tiling_upstream_bug(name):
    # strict: once upstream's fix is vendored this passes, and fails as an
    # unexpected pass until this xfail goes and fuzz/earcut.cpp asserts exact
    # tiling with many holes again
    sizes, text = UPSTREAM_HOLE_BUGS[name]
    points = np.frombuffer(base64.b64decode("".join(text)), dtype="<f8").reshape((-1, 2))
    rings = np.split(points, np.cumsum(sizes)[:-1])
    exterior, *holes = (np.ascontiguousarray(ring) for ring in rings)
    vertices, faces = trimkit.triangulate(exterior, holes)

    def ring_area(ring):
        x, y = ring[:, 0], ring[:, 1]
        return abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))) / 2

    expected = ring_area(exterior) - sum(ring_area(hole) for hole in holes)
    assert np.isclose(np.abs(areas2d(vertices, faces)).sum(), expected, rtol=0, atol=1e-9)
