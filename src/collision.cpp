// Collision and distance queries between rigidly placed triangle meshes, with
// coal-library/coal.
//
// A `CollisionScene` builds each of its meshes once: an OBBRSS bounding volume
// hierarchy over its triangles and, when it is declared convex, a coal
// convex shape whose support walks the hull. It only holds meshes: every
// query carries its own instances (which mesh, placed where), so nothing
// about a scene ever changes after it is built, and threads may share one
// freely.
//
// A pair of instances goes through one of two kernels: GJK/EPA between the
// convex shapes when both meshes are convex, which gives a signed distance
// and a normal, and coal's triangle BVH traversal otherwise. A pair whose
// world-space bounding boxes are apart is never handed to either. Only coal's
// `RelativeTransformationIsIdentity == 0` traversal nodes are used: the
// default ones rewrite the shared mesh's vertices into the other's frame.

#include "common.hpp"

#include <coal/BVH/BVH_model.h>
#include <coal/broadphase/broadphase_callbacks.h>
#include <coal/broadphase/broadphase_dynamic_AABB_tree_array.h>
#include <coal/collision_data.h>
#include <coal/collision_object.h>
#include <coal/internal/shape_shape_func.h>
#include <coal/internal/traversal_node_bvh_shape.h>
#include <coal/internal/traversal_node_bvhs.h>
#include <coal/internal/traversal_node_setup.h>
#include <coal/narrowphase/narrowphase.h>
#include <coal/shape/convex.h>
// coal's sources/: collide and distance over a traversal node
#include <collision_node.h>

#include <nanobind/stl/tuple.h>
#include <nanobind/stl/variant.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

namespace {

using MatricesArg = trimkit::Array<double, -1, 4, 4>;
using IndexArg = trimkit::Array<std::int64_t, -1>;
// every mesh convex or none, or a mask of which
using ConvexArg = std::variant<trimkit::Flag, trimkit::Array<bool, -1>>;
using PairsArg = trimkit::Array<std::int64_t, -1, 2>;
using Model = coal::BVHModel<coal::OBBRSS>;
using Convex = coal::ConvexTpl<coal::Triangle32>;
using MeshConvexCollision = coal::MeshShapeCollisionTraversalNode<coal::OBBRSS, coal::ConvexBase32, 0>;
using MeshConvexDistance = coal::MeshShapeDistanceTraversalNodeOBBRSS<coal::ConvexBase32>;
using Distances = std::tuple<trimkit::Out<double>, trimkit::Out<double>, trimkit::Out<double>>;

// a dropped template argument would select the overloads that write to the
// shared model, so pin the read-only ones here (the OBBRSS mesh/convex
// distance node is set up oriented, from a const model, by construction)
static_assert(coal::MeshCollisionTraversalNodeOBBRSS::RTIsIdentity == 0, "collision must not move the model");
static_assert(coal::MeshDistanceTraversalNodeOBBRSS::RTIsIdentity == 0, "distance must not move the model");
static_assert(MeshConvexCollision::RTIsIdentity == 0, "collision must not move the model");

// How far a rigid transform's rotation may be from orthonormal: matrices
// assembled in double precision are within ~1e-15, and float32 ones ~1e-7
constexpr double ORTHONORMAL_TOLERANCE = 1e-6;

double dot(const double* a, const double* b) { return (a[0] * b[0]) + (a[1] * b[1]) + (a[2] * b[2]); }

// One mesh of a CollisionScene, built once however often its queries place it
class CollisionMesh {
public:
    CollisionMesh(const trimkit::Mesh<std::uint32_t>& mesh, bool convex, const std::string& name)
        : face_count_(mesh.faces.size() / 3) {
        // a convex mesh is taken at its word: coal measures it as a solid,
        // walking its faces' edges to the vertex farthest along each
        // direction, so it needs faces, but whether they bound a convex
        // solid is the caller's promise to keep (an untrue one gives wrong
        // distances, never a crash)
        if (convex && face_count_ == 0) {
            trimkit::value_error(name + " is convex, so it needs faces");
        }
        if (face_count_ > 0) {
            const nb::gil_scoped_release release;
            build(mesh.vertices.data(), mesh.vertices.shape(0), mesh.faces, convex);
        }
    }

    std::size_t size() const { return face_count_; }
    bool empty() const { return face_count_ == 0; }
    const Model& model() const { return *model_; }
    const std::shared_ptr<Model>& shared_model() const { return model_; }
    const Convex* convex() const { return convex_.get(); }
    // the root of its hierarchy, the oriented box of every triangle
    const coal::OBB& bounds() const { return model_->getBV(0).bv.obb; }
    // the root's oriented box and rectangle swept sphere, both, as coal's
    // traversals test them
    const coal::OBBRSS& root() const { return model_->getBV(0).bv; }
    // the box of its vertices: (center, half extent) per axis
    const std::array<double, 6>& box() const { return box_; }

private:
    void build(const double* points, std::size_t vertex_count, const std::vector<std::uint32_t>& faces, bool convex) {
        // in coal's layout for addSubModel to copy into the model, which then
        // holds the only copy kept: the convex shape shares the model's
        std::vector<coal::Vec3s> vertices(vertex_count);
        for (std::size_t v = 0; v < vertex_count; v++) {
            vertices[v] = coal::Vec3s(points[3 * v], points[(3 * v) + 1], points[(3 * v) + 2]);
        }
        std::vector<coal::Triangle32> triangles(face_count_);
        for (std::size_t f = 0; f < face_count_; f++) {
            triangles[f] = coal::Triangle32(faces[3 * f], faces[(3 * f) + 1], faces[(3 * f) + 2]);
        }
        const trimkit::PointBounds bounds = trimkit::bounds_of(points, vertex_count);
        for (std::size_t axis = 0; axis < 3; axis++) {
            // halved before subtracting, so no extent overflows
            box_[axis] = (bounds.lowest[axis] / 2.0) + (bounds.highest[axis] / 2.0);
            box_[3 + axis] = (bounds.highest[axis] / 2.0) - (bounds.lowest[axis] / 2.0);
        }
        model_ = std::make_shared<Model>();
        model_->beginModel(static_cast<unsigned int>(face_count_), static_cast<unsigned int>(vertex_count));
        model_->addSubModel(vertices, triangles);
        model_->endModel();
        if (convex) {
            convex_ = std::make_unique<Convex>(model_->vertices, static_cast<unsigned int>(vertex_count),
                                               model_->tri_indices, static_cast<unsigned int>(face_count_));
        }
    }

    std::size_t face_count_ = 0;
    std::array<double, 6> box_{};
    std::shared_ptr<Model> model_;
    std::unique_ptr<Convex> convex_;
};

// An instance as one query places it: its mesh, rigid transform and world
// bounding box (center, half extent per axis)
struct Placed {
    const CollisionMesh* mesh = nullptr;
    coal::Transform3s transform;
    std::array<double, 6> box{};
};

// the gap between two boxes' surfaces, or 0 when they touch or overlap
double box_gap(const std::array<double, 6>& a, const std::array<double, 6>& b) {
    double squared = 0.0;
    for (std::size_t axis = 0; axis < 3; axis++) {
        const double apart = std::abs(a[axis] - b[axis]) - a[3 + axis] - b[3 + axis];
        squared += apart > 0.0 ? apart * apart : 0.0;
    }
    return std::sqrt(squared);
}

// b's pose in a's frame, as coal's traversals carry their second model
void relative(const Placed& a, const Placed& b, coal::Matrix3s& rotation, coal::Vec3s& translation) {
    rotation.noalias() = a.transform.getRotation().transpose() * b.transform.getRotation();
    translation.noalias() =
        a.transform.getRotation().transpose() * (b.transform.getTranslation() - a.transform.getTranslation());
}

// Whether two placed meshes may touch or come within `request`'s margin:
// their world boxes, then their hierarchies' roots as coal's collision
// traversal prunes, by oriented boxes. False only when they can't
bool may_touch(const Placed& a, const Placed& b, const coal::CollisionRequest& request) {
    if (a.mesh->empty() || b.mesh->empty() || box_gap(a.box, b.box) > request.security_margin) {
        return false;
    }
    // exactly as coal's (patched) mesh/mesh collision node tests its BVs: a
    // in b's frame, by the inverse of b's pose in a's
    coal::Matrix3s rotation;
    coal::Vec3s translation;
    relative(a, b, rotation, translation);
    const coal::Matrix3s rotation_transpose = rotation.transpose();
    const coal::Vec3s inverse_translation = -(rotation_transpose * translation);
    coal::Scalar lower = 0;
    return coal::overlapPrecomputedRTranspose(rotation_transpose, inverse_translation, b.mesh->root(), a.mesh->root(),
                                              request, lower);
}

// Whether two placed meshes may be at most `bound` apart: their world
// boxes, then their roots' rectangle swept spheres, a lower bound on their
// distance, as coal's distance traversal prunes. False only when they can't
bool may_be_within(const Placed& a, const Placed& b, double bound) {
    if (a.mesh->empty() || b.mesh->empty() || box_gap(a.box, b.box) > bound) {
        return false;
    }
    if (std::isinf(bound)) {
        return true;
    }
    coal::Matrix3s rotation;
    coal::Vec3s translation;
    relative(a, b, rotation, translation);
    return coal::distance(rotation, translation, a.mesh->root(), b.mesh->root()) <= bound;
}

// coal warm-starts every GJK from the solver's last run, its support search
// always (`set` keeps that), so each pair starts it afresh: no answer may
// depend on the pair before it, nor on how pairs fell across threads
template <typename Request>
void start_solver(coal::GJKSolver& solver, const Request& request) {
    solver.set(request);
    solver.cached_guess = coal::Vec3s(1, 0, 0);
    solver.support_func_cached_guess = coal::support_func_guess_t::Zero();
}

// Whether two placed meshes touch (or come within the request's margin):
// GJK/EPA between two convex solids, a mesh's triangles against a convex
// solid, or triangles against triangles. `result` is the caller's, reused.
bool collide_pair(const Placed& a, const Placed& b, const coal::CollisionRequest& request, coal::GJKSolver& solver,
                  coal::CollisionResult& result) {
    result.clear();
    const Convex* solid_a = a.mesh->convex();
    const Convex* solid_b = b.mesh->convex();
    if (solid_a != nullptr && solid_b != nullptr) {
        start_solver(solver, request);
        return coal::ShapeShapeCollide<coal::ConvexBase32, coal::ConvexBase32>(
                   solid_a, a.transform, solid_b, b.transform, &solver, request, result) > 0;
    }
    if (solid_a != nullptr || solid_b != nullptr) {
        const Placed& surface = solid_a != nullptr ? b : a;
        const Placed& solid = solid_a != nullptr ? a : b;
        start_solver(solver, request);
        MeshConvexCollision node(request);
        coal::initialize(node, surface.mesh->model(), surface.transform, solid.mesh->convex()->base(), solid.transform,
                         &solver, result);
        coal::collide(&node, request, result);
        return result.isCollision();
    }
    coal::MeshCollisionTraversalNodeOBBRSS node(request);
    coal::initialize(node, a.mesh->model(), a.transform, b.mesh->model(), b.transform, result);
    coal::collide(&node, request, result);
    return result.isCollision();
}

// a pair farther apart than asked about: infinitely far, nowhere
void write_far(double* distance, double* points, double* normal) {
    constexpr double NOTHING = std::numeric_limits<double>::quiet_NaN();
    *distance = std::numeric_limits<double>::infinity();
    std::fill(points, points + 6, NOTHING);
    std::fill(normal, normal + 3, NOTHING);
}

// The distance between two placed meshes, its closest points (a's, then b's)
// and the unit direction from a toward b. Signed when a convex solid is one
// side; between surfaces 0 once they touch, with no direction then. A pair
// farther apart than `upper_bound` is left as it is, infinitely far with no
// points or direction, and coal stops measuring it as soon as it can tell:
// GJK by its own early stop, a hierarchy by never descending into boxes
// farther than the closest distance so far, which starts at the bound.
void distance_pair(const Placed& a, const Placed& b, const coal::DistanceRequest& request, double upper_bound,
                   coal::GJKSolver& solver, coal::DistanceResult& result, double* distance, double* points,
                   double* normal) {
    // just past the bound, so a pair exactly that far is still measured
    const double past = std::isinf(upper_bound) ? std::numeric_limits<double>::max()
                                                : std::nextafter(upper_bound, std::numeric_limits<double>::infinity());
    result.clear();
    const Convex* solid_a = a.mesh->convex();
    const Convex* solid_b = b.mesh->convex();
    // coal measures from its first shape: a, unless a convex a goes second
    bool swapped = false;
    double measured = 0.0;
    if (solid_a != nullptr && solid_b != nullptr) {
        start_solver(solver, request);
        solver.distance_upper_bound = past;
        measured = coal::ShapeShapeDistance<coal::ConvexBase32, coal::ConvexBase32>(
            solid_a, a.transform, solid_b, b.transform, &solver, request, result);
        if (solver.gjk.status == coal::details::GJK::NoCollisionEarlyStopped) {
            write_far(distance, points, normal);  // proven farther than the bound
            return;
        }
    } else if (solid_a != nullptr || solid_b != nullptr) {
        swapped = solid_a != nullptr;
        const Placed& surface = swapped ? b : a;
        const Placed& solid = swapped ? a : b;
        start_solver(solver, request);
        result.min_distance = past;
        MeshConvexDistance node;
        coal::initialize(node, surface.mesh->model(), surface.transform, solid.mesh->convex()->base(), solid.transform,
                         &solver, request, result);
        coal::distance(&node);
        measured = result.min_distance;
    } else {
        result.min_distance = past;
        coal::MeshDistanceTraversalNodeOBBRSS node;
        coal::initialize(node, a.mesh->model(), a.transform, b.mesh->model(), b.transform, request, result);
        coal::distance(&node);
        measured = result.min_distance;
        // coal leaves two surfaces' normal unset: it is the direction between
        // the closest points
        result.normal = result.nearest_points[1] - result.nearest_points[0];
        result.normal = measured > 0.0 ? coal::Vec3s(result.normal / measured)
                                       : coal::Vec3s::Constant(std::numeric_limits<double>::quiet_NaN());
    }
    if (measured > upper_bound) {
        write_far(distance, points, normal);
        return;
    }
    *distance = measured;
    const double flip = swapped ? -1.0 : 1.0;
    for (Eigen::Index axis = 0; axis < 3; axis++) {
        points[axis] = result.nearest_points[swapped ? 1 : 0][axis];
        points[3 + axis] = result.nearest_points[swapped ? 0 : 1][axis];
        normal[axis] = flip * result.normal[axis];
    }
}

// The instances of pair `p`: row p of `rows`, or with no rows, the p-th
// i < j in numpy.triu_indices order, where row i of those starts at starts[i]
void pair_at(const std::int64_t* rows, const std::vector<std::size_t>& starts, std::size_t p, std::size_t* first,
             std::size_t* second) {
    if (rows != nullptr) {
        *first = static_cast<std::size_t>(rows[2 * p]);
        *second = static_cast<std::size_t>(rows[(2 * p) + 1]);
        return;
    }
    // the last row starting at or before p (rows that hold no pairs start
    // where the next does, and upper_bound passes them)
    const auto row = std::upper_bound(starts.begin(), starts.end(), p) - starts.begin() - 1;
    *first = static_cast<std::size_t>(row);
    *second = *first + 1 + (p - starts[*first]);
}

// collects every pair of objects whose boxes overlap, as (lower, higher) instance rows
struct Candidates final : coal::CollisionCallBackBase {
    std::vector<std::int64_t> pairs;
    bool collide(coal::CollisionObject* a, coal::CollisionObject* b) override {
        // each object's user data is its instance's row, a real int64
        const std::int64_t i = *static_cast<const std::int64_t*>(a->getUserData());
        const std::int64_t j = *static_cast<const std::int64_t*>(b->getUserData());
        pairs.push_back(std::min(i, j));
        pairs.push_back(std::max(i, j));
        return false;  // keep going: every candidate is wanted
    }
};

class CollisionScene {
public:
    // it owns its meshes (std::vector of std::unique_ptr claims to be
    // copyable to nanobind's traits, so say it isn't)
    CollisionScene(const CollisionScene&) = delete;
    CollisionScene& operator=(const CollisionScene&) = delete;

    CollisionScene(const trimkit::Meshes& meshes, const ConvexArg& convex_argument) {
        std::vector<trimkit::Mesh<std::uint32_t>> read;
        for (const nb::handle item : meshes) {
            read.push_back(trimkit::read_mesh<std::uint32_t>(item, "meshes[" + std::to_string(read.size()) + "]"));
        }
        std::vector<std::uint8_t> solid(read.size(), 0);
        if (const trimkit::Flag* every = std::get_if<trimkit::Flag>(&convex_argument)) {
            std::fill(solid.begin(), solid.end(), every->value ? 1 : 0);
        } else {
            const trimkit::In<std::uint8_t, -1> convex = trimkit::read(std::get<1>(convex_argument), "convex");
            if (convex.shape(0) != read.size()) {
                trimkit::value_error("convex must have one row per mesh");
            }
            std::copy(convex.data(), convex.data() + read.size(), solid.begin());
        }
        for (std::size_t i = 0; i < read.size(); i++) {
            meshes_.push_back(
                std::make_unique<CollisionMesh>(read[i], solid[i] != 0, "meshes[" + std::to_string(i) + "]"));
        }
    }

    std::size_t size() const { return meshes_.size(); }

    trimkit::Out<std::int64_t> collides(const MatricesArg& matrices_argument, const IndexArg& index_argument,
                                        const PairsArg& pairs_argument, trimkit::Number<double> margin_argument) const {
        const double margin = margin_argument;
        // NaN fails both comparisons, so it is refused too
        if (margin < 0.0 || margin > trimkit::COORDINATE_LIMIT || std::isnan(margin)) {
            trimkit::value_error(std::string("margin must be at least 0 and within ") + trimkit::COORDINATE_LIMIT_TEXT);
        }
        const std::vector<Placed> placed = place(matrices_argument, index_argument);
        // the pairs to test: the given rows, in place, or the boxes' candidates
        std::vector<std::int64_t> candidates;
        trimkit::In<std::int64_t, -1, 2> given;
        const std::int64_t* rows = nullptr;
        std::size_t pair_count = 0;
        if (pairs_argument.is_none()) {
            const nb::gil_scoped_release release;
            candidates = overlapping(placed, margin);
            rows = candidates.data();
            pair_count = candidates.size() / 2;
        } else {
            given = read_pairs(pairs_argument, placed.size());
            rows = given.data();
            pair_count = given.shape(0);
        }
        coal::CollisionRequest request(coal::NO_REQUEST, 1);
        request.security_margin = margin;
        // one pass: each pair culled, then tested, keeping those that touch
        std::vector<std::int64_t> touching = trimkit::gather_in_parallel<std::int64_t>(
            pair_count, [&](std::size_t begin, std::size_t end, std::vector<std::int64_t>& found) {
                coal::GJKSolver solver;
                coal::CollisionResult result;
                for (std::size_t p = begin; p < end; p++) {
                    const Placed& a = placed[static_cast<std::size_t>(rows[2 * p])];
                    const Placed& b = placed[static_cast<std::size_t>(rows[(2 * p) + 1])];
                    if (may_touch(a, b, request) && collide_pair(a, b, request, solver, result)) {
                        found.push_back(rows[2 * p]);
                        found.push_back(rows[(2 * p) + 1]);
                    }
                }
            });
        const std::size_t touching_count = touching.size() / 2;
        return trimkit::to_numpy(std::move(touching), {touching_count, 2});
    }

    Distances distances(const MatricesArg& matrices_argument, const IndexArg& index_argument,
                        const PairsArg& pairs_argument, trimkit::Number<double> upper_bound_argument) const {
        const double upper_bound = upper_bound_argument;
        if (!(upper_bound > 0.0)) {
            trimkit::value_error("upper_bound must be positive");
        }
        const std::vector<Placed> placed = place(matrices_argument, index_argument);
        // every pair is measured, so every pair is named: the given rows, in
        // place, or each i < j in numpy.triu_indices order, found from where
        // each row of them starts rather than listed
        const std::int64_t* rows = nullptr;
        std::vector<std::size_t> starts(placed.size());
        std::size_t pair_count = 0;
        trimkit::In<std::int64_t, -1, 2> given;
        if (pairs_argument.is_none()) {
            for (std::size_t i = 0; i < placed.size(); i++) {
                starts[i] = pair_count;
                pair_count += placed.size() - 1 - i;
            }
        } else {
            given = read_pairs(pairs_argument, placed.size());
            rows = given.data();
            pair_count = given.shape(0);
        }
        // every row written once, by its pair, so none is filled first
        std::vector<trimkit::Unset<double>> distance(pair_count);
        std::vector<trimkit::Unset<double>> points(6 * pair_count);
        std::vector<trimkit::Unset<double>> normals(3 * pair_count);
        double* distance_out = trimkit::elements(distance);
        double* points_out = trimkit::elements(points);
        double* normals_out = trimkit::elements(normals);
        const coal::DistanceRequest request;
        // one pass: each pair culled, then measured
        trimkit::in_parallel(pair_count, [&](std::size_t begin, std::size_t end) {
            coal::GJKSolver solver;
            coal::DistanceResult result;
            for (std::size_t p = begin; p < end; p++) {
                std::size_t first = 0;
                std::size_t second = 0;
                pair_at(rows, starts, p, &first, &second);
                const Placed& a = placed[first];
                const Placed& b = placed[second];
                if (may_be_within(a, b, upper_bound)) {
                    distance_pair(a, b, request, upper_bound, solver, result, &distance_out[p], &points_out[6 * p],
                                  &normals_out[3 * p]);
                } else {
                    write_far(&distance_out[p], &points_out[6 * p], &normals_out[3 * p]);
                }
            }
        });
        return {trimkit::to_numpy<double>(std::move(distance), {pair_count}),
                trimkit::to_numpy<double>(std::move(points), {pair_count, 2, 3}),
                trimkit::to_numpy<double>(std::move(normals), {pair_count, 3})};
    }

    // the box each instance is culled by, as these instances place it
    trimkit::BoxesObject boxes(const MatricesArg& matrices_argument, const IndexArg& index_argument) const {
        const std::vector<Placed> placed = place(matrices_argument, index_argument);
        std::vector<double> boxes(6 * placed.size());
        for (std::size_t i = 0; i < placed.size(); i++) {
            for (std::size_t axis = 0; axis < 3; axis++) {
                boxes[(6 * i) + axis] = placed[i].box[axis] - placed[i].box[3 + axis];
                boxes[(6 * i) + 3 + axis] = placed[i].box[axis] + placed[i].box[3 + axis];
            }
        }
        return trimkit::scene_boxes(boxes);
    }

private:
    // Which mesh each instance places and where, checked: `instance_index`
    // and `instance_matrices` may each be None, for one instance per mesh and
    // the identity. Each box bounds the mesh twice, carried into the scene
    // by rotating each center and taking half extents through the absolute
    // values of the rotated axes (Arvo): from the vertices' box, and from the
    // hierarchy's root oriented box. Either bounds it, so their overlap does:
    // the oriented box is far tighter for a mesh lying diagonally in its own
    // frame, and looser for a cube, whose principal axes are arbitrary
    std::vector<Placed> place(const MatricesArg& matrices_argument, const IndexArg& index_argument) const {
        const trimkit::Instances instances = trimkit::read_instances(index_argument, matrices_argument, meshes_.size());
        std::vector<Placed> placed(instances.count);
        // read and checked: the rest touches no Python
        const nb::gil_scoped_release release;
        for (std::size_t i = 0; i < instances.count; i++) {
            const double* m = instances.matrix(i);
            // a rotation: orthonormal rows, and no reflection
            bool rigid = true;
            for (std::size_t r = 0; r < 3; r++) {
                for (std::size_t c = 0; c < 3; c++) {
                    const double product = dot(m + (4 * r), m + (4 * c));
                    const double expected = r == c ? 1.0 : 0.0;
                    rigid = rigid && std::abs(product - expected) <= ORTHONORMAL_TOLERANCE;
                }
            }
            if (!rigid || trimkit::determinant_of(m) <= 0.0) {
                trimkit::value_error("instance_matrices must be rigid: instance " + std::to_string(i) +
                                     " rotates by a matrix that isn't a rotation");
            }
            const CollisionMesh* mesh = meshes_[instances.mesh(i)].get();
            coal::Matrix3s rotation;
            coal::Vec3s translation;
            for (Eigen::Index r = 0; r < 3; r++) {
                for (Eigen::Index c = 0; c < 3; c++) {
                    rotation(r, c) = m[(4 * r) + c];
                }
                translation[r] = m[(4 * r) + 3];
            }
            // a mesh without faces never touches anything: a point where it is placed
            std::array<double, 6> box{translation[0], translation[1], translation[2], 0.0, 0.0, 0.0};
            if (!mesh->empty()) {
                const coal::OBB& oriented = mesh->bounds();
                const std::array<double, 6>& vertex_box = mesh->box();
                const coal::Vec3s oriented_center = (rotation * oriented.To) + translation;
                const coal::Matrix3s oriented_axes = rotation * oriented.axes;
                const coal::Vec3s vertex_center =
                    (rotation * coal::Vec3s(vertex_box[0], vertex_box[1], vertex_box[2])) + translation;
                const coal::Vec3s vertex_half(vertex_box[3], vertex_box[4], vertex_box[5]);
                for (Eigen::Index r = 0; r < 3; r++) {
                    const double oriented_half = oriented_axes.row(r).cwiseAbs().dot(oriented.extent);
                    const double vertex_extent = rotation.row(r).cwiseAbs().dot(vertex_half);
                    const double lowest =
                        std::max(oriented_center[r] - oriented_half, vertex_center[r] - vertex_extent);
                    const double highest =
                        std::min(oriented_center[r] + oriented_half, vertex_center[r] + vertex_extent);
                    box[static_cast<std::size_t>(r)] = (lowest / 2.0) + (highest / 2.0);
                    box[3 + static_cast<std::size_t>(r)] = std::max(0.0, (highest / 2.0) - (lowest / 2.0));
                }
            }
            placed[i] = {mesh, coal::Transform3s(rotation, translation), box};
        }
        return placed;
    }

    // the instance pairs a query names, checked and read in place
    static trimkit::In<std::int64_t, -1, 2> read_pairs(const PairsArg& pairs_argument, std::size_t count) {
        const trimkit::In<std::int64_t, -1, 2> pairs = trimkit::read(pairs_argument, "pairs");
        const std::int64_t* rows = pairs.data();
        for (std::size_t p = 0; p < pairs.shape(0); p++) {
            const std::int64_t a = rows[2 * p];
            const std::int64_t b = rows[(2 * p) + 1];
            if (a < 0 || b < 0 || static_cast<std::size_t>(a) >= count || static_cast<std::size_t>(b) >= count) {
                trimkit::value_error("pairs names an instance out of range at row " + std::to_string(p));
            }
            if (a == b) {
                trimkit::value_error("pairs names one instance twice at row " + std::to_string(p));
            }
        }
        return pairs;
    }

    // Every pair of instances whose boxes come within `margin`, as ascending
    // (i < j) rows: coal's dynamic AABB tree, built for this query alone over
    // the placed boxes (each grown by half the margin), so nothing about the
    // scene changes and threads never share it
    static std::vector<std::int64_t> overlapping(const std::vector<Placed>& placed, double margin) {
        std::vector<coal::CollisionObject> objects;
        objects.reserve(placed.size());
        std::vector<coal::CollisionObject*> registered;
        registered.reserve(placed.size());
        std::vector<std::int64_t> rows_of(placed.size());
        std::iota(rows_of.begin(), rows_of.end(), std::int64_t{0});
        for (std::size_t i = 0; i < placed.size(); i++) {
            if (placed[i].mesh->empty()) {
                continue;  // a mesh without faces never touches anything
            }
            // the geometry's own box is never computed: these are the boxes
            coal::CollisionObject& object =
                objects.emplace_back(placed[i].mesh->shared_model(), placed[i].transform, false);
            object.setUserData(&rows_of[i]);
            const std::array<double, 6>& box = placed[i].box;
            for (Eigen::Index axis = 0; axis < 3; axis++) {
                const double half = box[3 + static_cast<std::size_t>(axis)] + (margin / 2.0);
                object.getAABB().min_[axis] = box[static_cast<std::size_t>(axis)] - half;
                object.getAABB().max_[axis] = box[static_cast<std::size_t>(axis)] + half;
            }
            registered.push_back(&object);
        }
        Candidates candidates;
        if (registered.size() > 1) {
            coal::DynamicAABBTreeArrayCollisionManager tree;
            tree.registerObjects(registered);
            tree.setup();
            tree.collide(&candidates);
        }
        // in ascending order, so answers never depend on the tree's shape,
        // and only pairs whose boxes are within the margin, as the explicit
        // pairs are held to
        std::vector<std::array<std::int64_t, 2>> rows(candidates.pairs.size() / 2);
        for (std::size_t p = 0; p < rows.size(); p++) {
            rows[p] = {candidates.pairs[2 * p], candidates.pairs[(2 * p) + 1]};
        }
        std::sort(rows.begin(), rows.end());
        std::vector<std::int64_t> kept;
        kept.reserve(candidates.pairs.size());
        for (const std::array<std::int64_t, 2>& row : rows) {
            if (box_gap(placed[static_cast<std::size_t>(row[0])].box, placed[static_cast<std::size_t>(row[1])].box) <=
                margin) {
                kept.push_back(row[0]);
                kept.push_back(row[1]);
            }
        }
        return kept;
    }

    std::vector<std::unique_ptr<CollisionMesh>> meshes_;
};

}  // namespace

void bind_collision(nb::module_& m) {
    nb::class_<CollisionScene>(m, "CollisionScene",
                               R"(Meshes to place and query against each other. A scene never changes:
each query names its own instances, so threads may share one.

Every query takes its instances (which mesh, placed where) and the pairs
of them to test, all optional, so a bare `scene.collides()` asks about the
scene as built: each mesh once, where it is, every pair. A robot passes its
links' transforms and the pairs of links not joined to each other.

`pairs` names exactly the pairs to test, which is how anything is left
out: a robot's ignored links, obstacles against each other. A path is one
query: N poses of L links are N * L instances, each pose's pairs named
(`pairs=None` would pair a link with itself at another pose), and the
pairs barely change from call to call, so build them once. Distances over
every pair are O(k**2) in memory. For gradients, use convex meshes: two
surfaces are 0 apart with no normal once they touch.

A convex mesh is tested as a solid and any other mesh as a surface, in
every pair: a mesh inside a convex one touches it, and a distance is
negative only when a convex solid is penetrated.

Parameters
----------
meshes : sequence of Mesh
  The meshes to place, each a (vertices, faces) tuple of (n, 3) float64
  vertices and (m, 3) int64 triangle indices into them. Copied once
  into coal's own storage, which the scene owns, so a mesh can never
  change under the threads sharing the scene.
convex : bool or (n,) bool, optional
  Which meshes to measure as convex solids, such as convex hulls:
  all of them or none, or a mask of which. Not checked: a mesh that
  isn't convex gives wrong distances, never a crash. Defaults to
  False.

Examples
--------
>>> import numpy as np
>>> import trimkit
>>> tetrahedron = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
>>> faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int64)
>>> scene = trimkit.CollisionScene([(tetrahedron, faces)], convex=True)
>>> len(scene)
1)")
        .def(nb::init<const trimkit::Meshes&, const ConvexArg&>(), nb::arg("meshes"),
             nb::arg("convex") = trimkit::Flag{})
        .def("__len__", &CollisionScene::size, R"(How many meshes the scene was built from.

Returns
-------
count : int
  The number of meshes the scene was built from.)")
        .def("boxes", &CollisionScene::boxes, nb::arg("instance_matrices").none() = nb::none(),
             nb::arg("instance_index").none() = nb::none(),
             R"(The box each instance is culled by, as a query with these instances
places it: what the scene thinks it holds.

Parameters
----------
instance_matrices : (k, 4, 4) float64, optional
  Each instance's rigid transform from its mesh into the scene.
  Defaults to the identity.
instance_index : (k,) int64, optional
  Which mesh each instance places. Defaults to each mesh once.

Returns
-------
boxes : Boxes
  One 3D box per instance, in its order, holding every face it places
  (an instance of a mesh without faces: a point at its translation).

Examples
--------
>>> import numpy as np
>>> import trimkit
>>> tetrahedron = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
>>> faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int64)
>>> scene = trimkit.CollisionScene([(tetrahedron, faces)], convex=True)
>>> placed = np.tile(np.eye(4), (2, 1, 1))
>>> scene.boxes().bounds.tolist()
[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])")
        .def("collides", &CollisionScene::collides, nb::arg("instance_matrices").none() = nb::none(),
             nb::arg("instance_index").none() = nb::none(), nb::kw_only(), nb::arg("pairs").none() = nb::none(),
             nb::arg("margin") = 0.0,
             R"(The pairs of instances that touch.

Parameters
----------
instance_matrices : (k, 4, 4) float64, optional
  Each instance's rigid transform from its mesh into the scene.
  Defaults to the identity.
instance_index : (k,) int64, optional
  Which mesh each instance places. Defaults to each mesh once.
pairs : (p, 2) int64, optional
  The pairs of instances to test. Defaults to every pair, found by
  their bounding boxes.
margin : float
  Also count pairs closer than this.

Returns
-------
pairs : (c, 2) int64
  Each pair of instances that touches or overlaps: rows of `pairs` in
  its order, or by default each touching i < j in ascending order.
  Empty when nothing touches.

Examples
--------
>>> import numpy as np
>>> import trimkit
>>> tetrahedron = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
>>> faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int64)
>>> scene = trimkit.CollisionScene([(tetrahedron, faces)], convex=True)
>>> placed = np.tile(np.eye(4), (2, 1, 1))
>>> placed[1, 0, 3] = 0.5
>>> scene.collides(placed, np.zeros(2, dtype=np.int64)).tolist()
[[0, 1]])")
        .def("distances", &CollisionScene::distances, nb::arg("instance_matrices").none() = nb::none(),
             nb::arg("instance_index").none() = nb::none(), nb::kw_only(), nb::arg("pairs").none() = nb::none(),
             nb::arg("upper_bound") = std::numeric_limits<double>::infinity(),
             R"(The distance between each pair of instances, and where it is.

Parameters
----------
instance_matrices : (k, 4, 4) float64, optional
  Each instance's rigid transform from its mesh into the scene.
  Defaults to the identity.
instance_index : (k,) int64, optional
  Which mesh each instance places. Defaults to each mesh once.
pairs : (p, 2) int64, optional
  The pairs of instances to measure. Defaults to every pair, in the
  order of `numpy.triu_indices(k, 1)`.
upper_bound : float
  Pairs farther apart than this are infinitely far, with no points or
  normal, and are measured only as far as it takes to tell.

Returns
-------
distances : (p,) float64
  The distance between each pair. A pair with a convex mesh has a signed
  distance, negative by how deep it reaches into the solid; two surfaces
  are 0 once their triangles touch.
points : (p, 2, 3) float64
  The closest point on each instance of a pair, in the scene.
normals : (p, 3) float64
  The unit direction from the first instance toward the second along
  which the distance is measured, or NaN for two surfaces that touch.

Examples
--------
>>> import numpy as np
>>> import trimkit
>>> tetrahedron = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
>>> faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int64)
>>> scene = trimkit.CollisionScene([(tetrahedron, faces)], convex=True)
>>> placed = np.tile(np.eye(4), (2, 1, 1))
>>> placed[1, 0, 3] = 2.0
>>> distances, points, normals = scene.distances(placed, np.zeros(2, dtype=np.int64))
>>> distances.tolist(), normals.tolist()
([1.0], [[1.0, 0.0, 0.0]]))");
}
