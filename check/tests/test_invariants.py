"""
The package-wide promises: numpy is the only dependency, Python is a
docstring, every array input is exactly typed and bounded, read in place when
C-contiguous and aligned and copied once otherwise, converted only when numpy
casts the dtype safely and then with a warning, every output is a fresh array
that owns its memory and aliases no input, and the GIL is released while
computing. Every public call is swept through every family of variants of
each argument its stub annotates (helpers.CASES, helpers.families), each call
checked for all of it and its answer against an independent oracle.
"""

import _thread
import collections
import contextlib
import copy
import functools
import gc
import importlib.metadata
import json
import os
import pathlib
import pickle
import sys
import threading
import time
import tracemalloc
import warnings

import numpy as np
import pytest
from helpers import (
    AS_GIVEN,
    CASES,
    LITTLE_ENDIAN,
    ORACLE,
    SIZES,
    SRC_BUGS,
    PoolCensus,
    assert_releases_gil,
    bound,
    box,
    call,
    checked_call,
    cpus_allowed,
    families,
    forked_answer,
    form_variants,
    heap_in_use,
    input_arrays,
    leaves,
    long_call,
    longest_gil_stall,
    public_api,
    raises_seen,
    run_python,
    runs_here,
    same_answers,
    sphere,
    stub_parameters,
)

import trimkit

API = public_api()
# the names with a slow call, and with a big one: the others' cases say why not
SLOW = [name for name, case in CASES.items() if callable(case.slow)]
BIG = [name for name, case in CASES.items() if callable(case.big)]


def test_cases_cover_the_api():
    # every public function, class and method has a case, and nothing else:
    # a new one can't go untested, and a renamed one can't linger. Each has an
    # oracle, each size a call or why there is none, every annotation of
    # every parameter a family generator, and every range a parameter's
    assert sorted(CASES) == API == sorted(stub_parameters())
    every = {p for parameters in stub_parameters().values() for p, _, _, _ in parameters}
    for name, case in CASES.items():
        assert callable(case.oracle) and callable(case.small), name
        for size in SIZES:
            assert callable(getattr(case, size)) or getattr(case, size), (name, size)
        assert set(case.ranges) <= every, name
        for _, annotation, _, _ in stub_parameters()[name]:
            assert families(annotation)


def test_an_annotation_without_a_generator_fails():
    with pytest.raises(TypeError, match="no family generates an argument annotated complex"):
        families("complex | None")


@pytest.mark.parametrize("name", API)
def test_every_family_of_every_argument(name):
    # the empty call, then every small form and every parameter of it
    # through every family its annotation has, each variant checked for
    # everything a call promises: answered as the form is, as the oracle
    # says, or refused as it must be. A family that never ran tested
    # nothing: every small form gives what it needs. Where SRC_BUGS names a
    # bug a variant must still fail, so a fix can't leave its line behind
    runs_here(name)
    case = CASES[name]
    census = PoolCensus()
    ran = collections.Counter()
    open_bugs = {key for key in SRC_BUGS if key.startswith(name + ": ")}
    if not isinstance(case.empty, str):
        checked_call(census, name, case.arguments("empty"))
    for number, form in enumerate(case.arguments("small")):
        answer = checked_call(census, name, form)
        for parameter, family, variant in form_variants(name, form):
            ran[parameter, family] += variant is not None
            if variant is None:
                continue
            expected = answer if variant.reference is AS_GIVEN else None
            if variant.reference is not AS_GIVEN and variant.reference is not ORACLE:
                expected = checked_call(census, name, {**form, parameter: variant.reference})
            given = {**form, parameter: variant.value}
            where = f"form {number}, {parameter} {family}"
            try:
                checked_call(census, name, given, variant=variant, expected=expected)
            except Exception as error:
                # a refusal returns, so this is a check failing: a bug known here, or new
                if f"{name}: {where}" not in SRC_BUGS:
                    error.add_note(f"{where}: {variant}")
                    raise
                open_bugs.discard(f"{name}: {where}")
                census.returned()
    idle = sorted(key for key, count in ran.items() if count == 0)
    assert not idle, f"families that never ran: {idle}"
    assert not open_bugs, f"these pass now, so SRC_BUGS can lose them: {sorted(open_bugs)}"


def pickled_objects() -> dict:
    """
    Constructor calls of every class the package binds, a full and an empty
    one each, by class name: keyword arguments in the canonical form a pickle
    gives them back in, and a query a pickled copy must answer exactly.
    """
    vertices, faces = sphere(16, 32)
    # a mesh without faces keeps vertices that no model holds
    loose = (np.ones((4, 3)), np.zeros((0, 3), dtype=np.int64))
    rays = np.random.default_rng(0).normal(size=(2, 200, 3))
    corners = np.random.default_rng(1).uniform(-1, 1, (300, 3))
    boxes = np.stack([corners, corners + 0.2], axis=1)
    meshes = [(vertices, faces), (vertices + 0.5, faces), loose]
    return {
        "Boxes": (
            [
                {"boxes": boxes},
                {"boxes": np.ascontiguousarray(boxes[:, :, :2])},
                {"boxes": np.zeros((0, 2, 3))},
            ],
            lambda tree: (
                tree.intersection(np.zeros((1, 2, tree.bounds.shape[1]))),
                *tree.nearest(np.zeros((1, 2, tree.bounds.shape[1])), maximum_hits=5),
            ),
        ),
        "RayScene": (
            [
                {"meshes": meshes, "use_gpu": False},
                {"meshes": [], "use_gpu": False},
                {"meshes": [], "use_gpu": True},
            ],
            lambda scene: scene.intersect(*rays),
        ),
        "CollisionScene": (
            [
                {"meshes": meshes, "convex": np.array([True, False, False])},
                {"meshes": [], "convex": np.zeros(0, dtype=bool)},
            ],
            lambda scene: scene.distances(),
        ),
        "CollisionResult": (
            [
                {
                    "pairs": np.array([[0, 1], [1, 2]], dtype=np.int64),
                    "faces": np.array([[3, -1], [-1, -1]], dtype=np.int64),
                    "distances": np.array([-0.25, np.inf]),
                    "points": np.array([[[1.0, 0.0, 0.0], [0.75, 0.0, 0.0]], [[np.nan] * 3, [np.nan] * 3]]),
                    "normals": np.array([[1.0, 0.0, 0.0], [np.nan] * 3]),
                },
                {
                    "pairs": np.array([[0, 1]], dtype=np.int64),
                    "faces": None,
                    "distances": None,
                    "points": None,
                    "normals": None,
                },
                {
                    "pairs": np.zeros((0, 2), dtype=np.int64),
                    "faces": None,
                    "distances": None,
                    "points": None,
                    "normals": None,
                },
            ],
            lambda result: result,
        ),
    }


def test_every_class_pickles():
    # every class the package binds pickles as exactly the keyword arguments
    # it was built with, only arrays and Python values, never a tree: a load
    # or a deep copy answers bit for bit what the original does
    classes = sorted(name for name in API if "." not in name and isinstance(getattr(trimkit, name), type))
    objects = pickled_objects()
    assert sorted(objects) == classes
    with warnings.catch_warnings():
        # a scene asked for a device answers on the CPU without one, saying so
        warnings.simplefilter("ignore", UserWarning)
        for name, (calls, query) in objects.items():
            for keywords in calls:
                instance = getattr(trimkit, name)(**keywords)
                constructor, positional = instance.__reduce__()
                assert constructor.func is type(instance) and constructor.args == () and positional == ()
                assert sorted(constructor.keywords) == sorted(keywords)
                for key, given in keywords.items():
                    # an array for an array, True, False and None themselves
                    assert same_answers(constructor.keywords[key], given), key
                data = pickle.dumps(instance)
                # a pickle this test just made, so trusted
                for loaded in (pickle.loads(data), copy.deepcopy(instance)):  # noqa: S301
                    assert type(loaded) is type(instance)
                    assert pickle.dumps(loaded) == data
                    # bit for bit, the NaNs a pair with a faceless mesh answers too
                    assert same_answers(query(loaded), query(instance))


@pytest.mark.timing
@pytest.mark.skipif(not getattr(sys, "_is_gil_enabled", lambda: True)(), reason="no GIL to hold")
def test_stall_detector_catches_a_held_gil():
    # the control: sorting a big list is one C call that never releases it,
    # held off for most of the call (the rest is the call's own setup).
    # Shuffled, as timsort sorts a run (a reversed range) in linear time
    data = np.random.default_rng(0).permutation(1_000_000).tolist()
    assert longest_gil_stall(lambda: sorted(data)) > 0.5
    # and the one test of release fails it, after every measurement
    with pytest.raises(AssertionError, match="the least of"):
        assert_releases_gil(lambda: sorted(data))


@pytest.mark.timing
@pytest.mark.parametrize("name", SLOW)
def test_releases_gil(name):
    runs_here(name)
    # only the method is measured: its object's constructor, built first,
    # would hold off the thread for none of the method's own time
    assert_releases_gil(bound(name, CASES[name].arguments("slow")))


def test_only_numpy():
    requires = importlib.metadata.requires("trimkit") or []
    assert [r for r in requires if "extra ==" not in r] == ["numpy>=2"]


def test_the_package_is_the_extension():
    # no Python at all: `trimkit` is one extension, the package's __init__,
    # beside only its stubs and typing marker
    root = pathlib.Path(trimkit.__file__).parent
    assert pathlib.Path(trimkit.__file__).name.startswith("__init__.")
    assert not pathlib.Path(trimkit.__file__).name.endswith(".py")
    assert sorted(
        p.name for p in root.iterdir() if p.name != "__pycache__" and p != pathlib.Path(trimkit.__file__)
    ) == [
        "__init__.pyi",
        "exchange.pyi",
        "py.typed",
    ]


def bytes_held(name: str) -> int:
    """
    What 50 more rounds of public call `name` leave allocated, answering
    and refusing, after 50 to warm it: counted by glibc, so every
    allocation is in it, ours, numpy's and the vendored libraries' alike
    """
    form = CASES[name].arguments("small")[0]
    # and refused: the first variant of its arguments that must be
    parameter, variant = next(
        (p, v) for p, _, v in form_variants(name, form) if v is not None and v.answers is False
    )
    calls = [form, {**form, parameter: variant.value}]

    def every_call():
        for given in calls:
            with contextlib.suppress(ValueError, TypeError):
                call(name, given)

    for _ in range(50):
        every_call()
    gc.collect()
    before = heap_in_use()
    for _ in range(50):
        every_call()
    gc.collect()
    return heap_in_use() - before


@functools.cache
def _held_by_every_call() -> dict[str, int]:
    # every call measured in one fresh process with glibc's per-thread cache
    # off and one arena: a pool worker lives on between calls, and the chunks
    # its cache keeps count as in use, though they're a few per size and per
    # thread, filling once, which no call holds. So do the arenas glibc adds
    # as workers contend, each a few kilobytes counted in use from its first
    # allocation: on 32 cores 10-60 KB appeared over 50 rounds of
    # Boxes.intersection at random, and none with one arena
    script = (
        "import json, sys\n"
        f"sys.path.insert(0, {str(pathlib.Path(__file__).parent)!r})\n"
        "import test_invariants as t\n"
        "print(json.dumps({name: t.bytes_held(name) for name in sys.argv[1:]}))\n"
    )
    environment = {**os.environ, "GLIBC_TUNABLES": "glibc.malloc.tcache_count=0:glibc.malloc.arena_max=1"}

    def runs(name: str) -> bool:
        try:
            runs_here(name)
        except pytest.skip.Exception:
            return False
        return True

    done = run_python(
        "-W",
        "ignore",
        "-c",
        script,
        *filter(runs, API),
        text=True,
        env=environment,
        check=True,
        timeout=600,
    )
    return json.loads(done.stdout.splitlines()[-1])


@pytest.mark.parametrize("name", API)
def test_calls_hold_no_memory(name):
    # What a call allocates it frees, whether it answers or refuses: many
    # rounds leave the heap where it was
    runs_here(name)
    if heap_in_use() is None:
        pytest.skip("the heap is counted by glibc's mallinfo2")
    grown = _held_by_every_call()[name]
    assert grown < 16 * 1024, f"{grown} bytes held after 50 more rounds"


def test_a_wrong_shape_is_refused_before_anything_converts():
    # the shape is checked first: no conversion warning (which this suite
    # raises) for an array that is refused anyway, whatever its dtype
    for wrong in (np.zeros((4, 3), dtype=np.float32), np.zeros((4, 3)), [[0, 0, 0], [1, 0, 0]], [0.0, 1.0]):
        with pytest.raises(TypeError, match="exterior"):
            trimkit.triangulate(wrong)
    with pytest.raises(TypeError, match="instance_index"):
        trimkit.RayScene([box()]).intersect_any(np.zeros(3), np.ones(3), None, [[0]])


def _peak_python_bytes(query) -> int:
    """
    The most memory Python and numpy held at once during `query`, above what
    they held before it: numpy reports every array buffer it allocates to
    tracemalloc, and our own C++ allocations and outputs don't go through it.
    """
    tracemalloc.start()
    try:
        before, _ = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
        query()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak - before


def test_copy_detector_catches_copies():
    # the control: a copy the size of the input is plainly visible
    data = np.random.default_rng(0).random((50_000, 3))
    assert _peak_python_bytes(lambda: np.ascontiguousarray(data[::-1])) >= data.nbytes


@pytest.mark.parametrize("name", BIG)
def test_inputs_are_never_copied(name):
    # nothing converts or copies an input on the way in: during the call
    # Python and numpy allocate far less than the largest input, beyond what
    # the answer is made of there (draco's bytes, a property's numpy copy);
    # a str is read in place too
    runs_here(name)
    given = CASES[name].arguments("big")
    texts = [len(value) for value in given.values() if isinstance(value, str)]
    largest = max([a.nbytes for a in input_arrays(given)] + texts)
    assert largest > 200_000
    answered = 0
    for leaf in leaves(call(name, given)):
        if isinstance(leaf, (bytes, str)) or (isinstance(leaf, np.ndarray) and leaf.flags.owndata):
            answered += leaf.nbytes if isinstance(leaf, np.ndarray) else len(leaf)
    assert _peak_python_bytes(lambda: call(name, given)) < largest // 2 + answered


@pytest.mark.timing
def test_ctrl_c_stops_long_queries():
    # every query polls for Ctrl-C with the GIL released, in the one place,
    # trimkit.hpp's in_parallel, polled through gil.hpp's with_pool, which no binding can go around (`scripts
    # static`), so one query tests them all: an interrupt 50 ms in stops it
    # well before it would finish. It takes a second here (helpers.long_call),
    # so this holds on fast and slow machines alike: stopping takes the 50 ms
    # timer plus up to one 50 ms poll period, and at 0.5 s a loaded machine
    # missed the half-call bound by 5 ms (2026-11-01)
    query, uninterrupted = long_call()
    timer = threading.Timer(0.05, _thread.interrupt_main)
    tic = time.perf_counter()
    timer.start()
    with pytest.raises(KeyboardInterrupt):
        query()
    timer.join()
    assert time.perf_counter() - tic < uninterrupted / 2


@pytest.mark.timing
@pytest.mark.parametrize("name", ["decompose", "parametrize"])
def test_ctrl_c_in_a_library_answers_whole_or_raises(name):
    # a vendored library polls only from its own progress reports, which
    # V-HACD makes between stages (its vertex welding, before any, is
    # seconds on a mesh of a million triangles) and xatlas only when a
    # stage's percentage changes: so how soon it stops isn't promised. What
    # is: Ctrl-C at any moment raises KeyboardInterrupt or the call answers
    # whole, never part of an answer, and the next call answers the same
    given = CASES[name].arguments("slow")
    tic = time.perf_counter()
    expected = call(name, given)
    took = time.perf_counter() - tic
    # delays within the call, until at least one interrupt landed inside it
    # (the library polled, and the call raised from inside), or nothing here
    # tested an interrupted call: one that never polls finishes, and the
    # interrupt is raised only after it returned
    landed = 0
    for trial, fraction in enumerate(np.random.default_rng(3).uniform(0.05, 0.6, 16)):
        if trial >= 4 and landed:
            break
        timer = threading.Timer(fraction * took, _thread.interrupt_main)
        timer.start()
        answer = None
        with raises_seen(getattr(trimkit, name)) as raised:
            try:
                answer = call(name, given)
            except KeyboardInterrupt:
                landed += bool(raised)
            else:
                # the call finished first: the interrupt lands here, between sleeps
                with contextlib.suppress(KeyboardInterrupt):
                    for _ in range(1000):
                        time.sleep(0.01)
        timer.join()
        assert answer is None or same_answers(answer, expected)
        assert same_answers(call(name, given), expected)
    assert landed, f"no interrupt landed inside {name}: nothing polled during it"


@pytest.mark.timing
def test_random_ctrl_c_leaves_the_pool_healthy():
    # Ctrl-C at random moments in a long parallel query, workers mid-range
    # or not: each raises promptly, and the next call is whole and the same
    query, uninterrupted = long_call()
    expected = query()
    # delays within the first half of a call of a second or more, so each
    # lands inside it
    for fraction in np.random.default_rng(7).uniform(0.005, 0.5, 6):
        delay = fraction * uninterrupted
        timer = threading.Timer(delay, _thread.interrupt_main)
        tic = time.perf_counter()
        timer.start()
        with pytest.raises(KeyboardInterrupt):
            query()
            # the call finished first: the interrupt lands here, between
            # sleeps, as interrupt_main sends no signal to cut one short
            for _ in range(1000):
                time.sleep(0.01)
        timer.join()
        # a 50 ms poll and one range at most: well before a call that
        # ignored it would have ended
        assert time.perf_counter() - tic < delay + (uninterrupted - delay) / 2
    assert np.array_equal(query(), expected)


# big enough that every parallel query wakes workers, hashed to compare runs
_PARALLEL_QUERIES = """
import hashlib, os, sys
import numpy as np
sys.path.insert(0, sys.argv[2])
from helpers import ball_runs, box, collision_fields, sphere
import trimkit
if sys.argv[1] == "one":
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
rng = np.random.default_rng(0)
corners = rng.uniform(0, 100, (50_000, 3))
queries = rng.uniform(0, 100, (50_000, 3))
scene = trimkit.RayScene([sphere(60, 120)])
origins, directions = rng.normal(size=(2, 100_000, 3)) * 0.1
solids = trimkit.CollisionScene([sphere(10, 20)], convex=True)
matrices = np.tile(np.eye(4), (300, 1, 1))
matrices[:, :3, 3] = rng.uniform(0, 15, (300, 3))
index = np.zeros(300, np.int64)
boxes = trimkit.Boxes(np.stack([corners, corners + 1.0], axis=1))
outputs = [
    boxes.intersection(np.stack([queries, queries + 3.0], axis=1)),
    *scene.intersect(origins, directions, maximum_hits=1),
    *scene.intersect(origins, directions),
    *collision_fields(solids.distances(matrices, index)),
    *collision_fields(solids.collides(matrices, index)),
    *collision_fields(solids.collides(matrices, index, return_distance=True, maximum_contacts=4)),
    # xatlas's tasks on however many threads, each with its own scratch
    *trimkit.parametrize(*sphere(30, 60)),
    # contour's passes, and V-HACD's fan-outs, on the pool
    *trimkit.contour(ball_runs(60)),
    *(array for hull in trimkit.decompose(*box(), resolution=20_000) for array in hull),
]
print(hashlib.sha256(b"".join(np.ascontiguousarray(a).tobytes() for a in outputs)).hexdigest())
"""


@pytest.mark.skipif(not hasattr(os, "sched_setaffinity"), reason="needs CPU affinity (Linux)")
def test_one_core_matches_every_core():
    # however many workers a query gets, down to none, the answers are the
    # same bytes: each item's output depends on that item alone
    if cpus_allowed() < 2:
        pytest.skip("one CPU: every core is the one core, so there is nothing to compare")
    tests = str(pathlib.Path(__file__).parent)
    digests = {
        cores: run_python(
            "-c",
            _PARALLEL_QUERIES,
            cores,
            tests,
            check=True,
            text=True,
        ).stdout
        for cores in ("one", "every")
    }
    assert digests["one"] == digests["every"] != ""


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs fork")
def test_fork_after_a_parallel_call():
    # a child forked while the parent's workers are alive starts its own and
    # answers the same; Python warns about forking with threads alive
    query, _ = long_call(0.1)
    # a child that hangs fails this rather than the whole run
    assert forked_answer(query, query(), 120.0) == b"1"


def test_enums_are_int_enums_named_for_their_function():
    # every enumeration is an IntEnum beside the function taking it, each
    # built from its int and an int itself, and named for it: DracoKind for
    # exchange.draco_*, SvgKind for exchange.svg_*, TriangulateEngine for triangulate
    import enum
    import re

    from trimkit.exchange import DracoKind

    exchange = trimkit.exchange
    assert DracoKind is exchange.DracoKind and sys.modules["trimkit.exchange"] is exchange
    found = []
    for module in (trimkit, exchange):
        families = [v for v in vars(module).values() if isinstance(v, type) and issubclass(v, enum.Enum)]
        functions = [n for n, v in vars(module).items() if callable(v) and not isinstance(v, type)]
        for family in families:
            found.append(family.__name__)
            assert issubclass(family, enum.IntEnum)
            for member in family:
                assert family(int(member)) is member
                assert member == int(member)
            prefix = re.match(r"[A-Z][a-z]+", family.__name__)[0].lower()
            assert [f for f in functions if f == prefix or f.startswith(prefix + "_")], family
    assert sorted(found) == ["DracoKind", "SvgKind", "TriangulateEngine"]
    assert int(exchange.SvgKind.CUBIC) == 2
    assert int(trimkit.TriangulateEngine.CDT) == 1
    assert exchange.SvgSegment.fields["kind"][0] == np.uint8


def test_draco_refuses_only_big_endian():
    # on a big-endian machine both draco calls refuse before reading anything
    # (every value would come out byte-swapped); anywhere else they work
    vertices, faces = box()
    kind = np.array([trimkit.exchange.DracoKind.POSITION], dtype=np.int64)
    position = [vertices.astype(np.float32)]
    if LITTLE_ENDIAN:
        buffer = trimkit.exchange.draco_encode(faces, position, kind)
        assert len(
            trimkit.exchange.draco_decode(np.frombuffer(buffer, np.uint8), np.array([0], dtype=np.int64))[0]
        ) == len(faces)
        return
    message = "draco reads and writes only little-endian data, and this machine is big-endian"
    with pytest.raises(NotImplementedError, match=message):
        trimkit.exchange.draco_encode(faces, position, kind)
    with pytest.raises(NotImplementedError, match=message):
        trimkit.exchange.draco_decode(np.zeros(8, np.uint8), np.array([0], dtype=np.int64))


def _public(name: str):
    # a public name of `trimkit`, `X` or `X.member`, as the object documented
    import trimkit

    if name == "trimkit":
        return trimkit
    if name == "trimkit.exchange":
        return trimkit.exchange
    owner, _, member = name.partition(".")
    value = getattr(trimkit, owner)
    return vars(value)[member] if member else value


@pytest.mark.parametrize("name", [*API, "trimkit", "trimkit.exchange"])
def test_examples_run(name):
    # every public callable's numpy Examples section runs and prints exactly
    # what it says: no example in a docstring or the README can rot
    import doctest

    runs_here(name)
    doc = _public(name).__doc__
    test = doctest.DocTestParser().get_doctest(doc, {}, name, None, 0)
    assert test.examples, f"{name} has no Examples section"
    runner = doctest.DocTestRunner(optionflags=doctest.NORMALIZE_WHITESPACE)
    runner.run(test)
    assert runner.failures == 0


def _attributes(doc: str) -> list[str]:
    # the names a numpy Attributes section lists
    section = doc.split("Attributes\n----------\n", 1)[1].split("\n\n", 1)[0]
    return [line.split(" :")[0] for line in section.splitlines() if line and not line[0].isspace()]


def test_docs_name_everything():
    # the module names its every non-callable public name, each enum its
    # every member (each with its own docstring), and each __len__ what it
    # counts: the numpy sections the README tables are built from
    import enum
    import types

    plain = [
        name
        for name in dir(trimkit)
        if not name.startswith("_")
        and (not callable(getattr(trimkit, name)) or isinstance(getattr(trimkit, name), types.GenericAlias))
    ]
    assert sorted(_attributes(trimkit.__doc__)) == sorted([*plain, "__version__"])
    assert _attributes(trimkit.exchange.__doc__) == ["DracoKind", "SvgKind", "SvgSegment"]
    families = [
        v
        for module in (trimkit, trimkit.exchange)
        for v in vars(module).values()
        if isinstance(v, type) and issubclass(v, enum.Enum)
    ]
    for family in families:
        assert _attributes(family.__doc__) == [m.name for m in family]
        assert all(m.__doc__ and m.__doc__ != family.__doc__ for m in family)
    for name in API:
        value = _public(name)
        if isinstance(value, type) and "__len__" in vars(value):
            assert "Returns\n-------\ncount : int" in value.__len__.__doc__, name


def test_version_is_the_packages():
    assert trimkit.__version__ == importlib.metadata.version("trimkit")
