// Collision and distance queries between rigidly placed triangle meshes, with
// coal-library/coal. Python-free (trimkit.hpp).
//
// A `CollisionMeshes` builds each of its meshes once: an OBBRSS bounding
// volume hierarchy over its triangles and, when it is declared convex, a coal
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

#include "trimkit.hpp"

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

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

using Model = coal::BVHModel<coal::OBBRSS>;
using Convex = coal::ConvexTpl<coal::Triangle32>;
using MeshConvexCollision = coal::MeshShapeCollisionTraversalNode<coal::OBBRSS, coal::ConvexBase32, 0>;
using MeshConvexDistance = coal::MeshShapeDistanceTraversalNodeOBBRSS<coal::ConvexBase32>;

// a dropped template argument would select the overloads that write to the
// shared model, so pin the read-only ones here (the OBBRSS mesh/convex
// distance node is set up oriented, from a const model, by construction)
static_assert(coal::MeshCollisionTraversalNodeOBBRSS::RTIsIdentity == 0, "collision must not move the model");
static_assert(coal::MeshDistanceTraversalNodeOBBRSS::RTIsIdentity == 0, "distance must not move the model");
static_assert(MeshConvexCollision::RTIsIdentity == 0, "collision must not move the model");

}  // namespace

// One mesh of a CollisionMeshes, built once however often its queries place it
class trimkit::CollisionMesh {
public:
    explicit CollisionMesh(const CollisionMeshInput& mesh) : face_count_(mesh.face_count) {
        if (face_count_ == 0) {
            // no model holds them, so a pickle can still give them back
            loose_vertices_.assign(mesh.vertices, mesh.vertices + (3 * mesh.vertex_count));
            return;
        }
        // in coal's layout for addSubModel to copy into the model, which then
        // holds the only copy kept: the convex shape shares the model's
        std::vector<coal::Vec3s> vertices(mesh.vertex_count);
        for (std::size_t v = 0; v < mesh.vertex_count; v++) {
            vertices[v] = coal::Vec3s(mesh.vertices[3 * v], mesh.vertices[(3 * v) + 1], mesh.vertices[(3 * v) + 2]);
        }
        std::vector<coal::Triangle32> triangles(face_count_);
        for (std::size_t f = 0; f < face_count_; f++) {
            triangles[f] = coal::Triangle32(mesh.faces[3 * f], mesh.faces[(3 * f) + 1], mesh.faces[(3 * f) + 2]);
        }
        bounds_ = bounds_of(mesh.vertices, mesh.vertex_count);
        model_ = std::make_shared<Model>();
        model_->beginModel(static_cast<unsigned int>(face_count_), static_cast<unsigned int>(mesh.vertex_count));
        model_->addSubModel(vertices, triangles);
        model_->endModel();
        if (mesh.convex) {
            convex_ = std::make_unique<Convex>(model_->vertices, static_cast<unsigned int>(mesh.vertex_count),
                                               model_->tri_indices, static_cast<unsigned int>(face_count_));
            convex_->computeLocalAABB();  // the center its GJK starts from
        }
    }

    // the mesh as it was given: the model's own copy, in the order given
    void as_given(std::vector<double>* vertices, std::vector<std::int64_t>* faces) const {
        *vertices = loose_vertices_;
        faces->assign(3 * face_count_, 0);
        if (model_) {
            vertices->resize(3 * model_->vertices->size());
            for (std::size_t i = 0; i < model_->vertices->size(); i++) {
                for (std::size_t axis = 0; axis < 3; axis++) {
                    (*vertices)[(3 * i) + axis] = (*model_->vertices)[i][static_cast<Eigen::Index>(axis)];
                }
            }
            for (std::size_t f = 0; f < face_count_; f++) {
                for (std::size_t corner = 0; corner < 3; corner++) {
                    (*faces)[(3 * f) + corner] = (*model_->tri_indices)[f][static_cast<std::uint32_t>(corner)];
                }
            }
        }
    }

    bool empty() const { return face_count_ == 0; }
    const Model& model() const { return *model_; }
    const std::shared_ptr<Model>& shared_model() const { return model_; }
    const Convex* convex() const { return convex_.get(); }
    // the root of its hierarchy, the oriented box of every triangle
    const coal::OBB& bounds() const { return model_->getBV(0).bv.obb; }
    // the root's oriented box and rectangle swept sphere, both, as coal's
    // traversals test them
    const coal::OBBRSS& root() const { return model_->getBV(0).bv; }
    // the box of its vertices, its least and greatest value on each axis
    const PointBounds& corners() const { return bounds_; }

private:
    std::size_t face_count_ = 0;
    // the vertices of a mesh without faces, which no model holds
    std::vector<double> loose_vertices_;
    PointBounds bounds_{};
    std::shared_ptr<Model> model_;
    std::unique_ptr<Convex> convex_;
};

namespace {

using trimkit::CollisionMesh;
using trimkit::CollisionRow;

// An instance as one query places it: its mesh, rigid transform and world
// bounding box (center, half extent per axis)
struct Placed {
    const CollisionMesh* mesh = nullptr;
    coal::Transform3s transform;
    std::array<double, 6> box{};     // (center, half extent) per axis, as the culls test it
    trimkit::PointBounds corners{};  // the same box as its two corners, exactly as computed
};

// what a cull pads a placed box by: BOX_SLACK of its farthest coordinate,
// so rounding in placing it never culls a pair coal's tests find touching
double slack(const std::array<double, 6>& box) {
    double farthest = 0.0;
    for (std::size_t axis = 0; axis < 3; axis++) {
        farthest = std::max(farthest, std::abs(box[axis]) + box[3 + axis]);
    }
    return trimkit::BOX_SLACK * farthest;
}

// a lower bound on the gap between two boxes' surfaces, each padded by its
// slack, or 0 when they touch or overlap
double box_gap(const std::array<double, 6>& a, const std::array<double, 6>& b) {
    const double padded = slack(a) + slack(b);
    double squared = 0.0;
    for (std::size_t axis = 0; axis < 3; axis++) {
        const double apart = std::abs(a[axis] - b[axis]) - a[3 + axis] - b[3 + axis] - padded;
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
// traversal prunes, by oriented boxes. False only when they can't. The root
// test is not coal's work done twice for convex solids, which have no
// hierarchy between the boxes and GJK: without it, three robots' named pairs
// took 6 to 11, 146 to 218 and 117 to 200 microseconds a configuration
// (2026-11-12, answers identical); between meshes it changes nothing
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

// How far a measurement of two placed meshes runs to settle whether they
// are within `bound`: past it by their boxes' slack, and by at least one
// double. Every lower bound that culls or stops a measurement early (the
// boxes', the roots', GJK's) rounds like a box does, so one held to the bare
// bound loses pairs exactly that far; whether a pair is within is decided
// after, exactly, on the distance measured (`distance_pair`)
double cutoff_for(const Placed& a, const Placed& b, double bound) {
    if (std::isinf(bound)) {
        return std::numeric_limits<double>::max();
    }
    return std::max(std::nextafter(bound, std::numeric_limits<double>::infinity()),
                    bound + slack(a.box) + slack(b.box));
}

// Whether two placed meshes may be within `cutoff_for` of them: their world
// boxes, then their roots' rectangle swept spheres, a lower bound on their
// distance, as coal's distance traversal prunes. False only when they can't.
// Worth up to 7% of a bounded query between convex solids, and nothing
// between meshes (2026-11-12, the same three robots, answers identical)
bool may_be_within(const Placed& a, const Placed& b, double cutoff) {
    if (a.mesh->empty() || b.mesh->empty() || box_gap(a.box, b.box) > cutoff) {
        return false;
    }
    if (cutoff == std::numeric_limits<double>::max()) {
        return true;
    }
    coal::Matrix3s rotation;
    coal::Vec3s translation;
    relative(a, b, rotation, translation);
    return coal::distance(rotation, translation, a.mesh->root(), b.mesh->root()) <= cutoff;
}

// coal warm-starts every support search from the solver's last run (`set`
// keeps that), so each pair starts it afresh: no answer may depend on the
// pair before it, nor on how pairs fell across threads
template <typename Request>
void start_solver(coal::GJKSolver& solver, const Request& request) {
    solver.set(request);
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
        // GJK starts between the two boxes' centers, as MuJoCo's between its
        // geoms', not along coal's (1, 0, 0): the pair's own data, so as
        // deterministic, and one support step tells most separated pairs
        // apart. Only here, where both shapes have a box (a triangle has none)
        solver.gjk_initial_guess = coal::GJKInitialGuess::BoundingVolumeGuess;
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
void write_far(CollisionRow* row) {
    constexpr double NOTHING = std::numeric_limits<double>::quiet_NaN();
    row->faces[0] = row->faces[1] = coal::Contact::NONE;
    row->distance = std::numeric_limits<double>::infinity();
    std::fill(row->points, row->points + 6, NOTHING);
    std::fill(row->normal, row->normal + 3, NOTHING);
}

// What coal measured or found for a pair, in the pair's (a, b) order: coal
// measures from its first shape, which is b when `swapped` (a convex a against
// a mesh b, whose surface goes first), so its faces and witness points trade
// places and its normal, from its first shape toward its second, turns around
void write_witness(bool swapped, std::int64_t first_face, std::int64_t second_face,
                   const std::array<coal::Vec3s, 2>& witnesses, const coal::Vec3s& normal, double distance,
                   CollisionRow* row) {
    row->faces[0] = swapped ? second_face : first_face;
    row->faces[1] = swapped ? first_face : second_face;
    row->distance = distance;
    const double flip = swapped ? -1.0 : 1.0;
    for (Eigen::Index axis = 0; axis < 3; axis++) {
        row->points[axis] = witnesses[swapped ? 1 : 0][axis];
        row->points[3 + axis] = witnesses[swapped ? 0 : 1][axis];
        row->normal[axis] = flip * normal[axis];
    }
}

// whether coal puts b first for this pair: a convex a against a mesh b
bool swapped_for(const Placed& a, const Placed& b) {
    return a.mesh->convex() != nullptr && b.mesh->convex() == nullptr;
}

// The distance between two placed meshes, its closest points (a's, then b's)
// and the unit direction from a toward b. Signed when a convex solid is one
// side; between surfaces 0 once they touch, with no direction then. A pair
// farther apart than `upper_bound` is left as it is, infinitely far with no
// points or direction, and coal stops measuring it as soon as it can tell:
// GJK by its own early stop, a hierarchy by never descending into boxes
// farther than the closest distance so far, which starts at `cutoff`
// (`cutoff_for` the bound): neither stops on a pair exactly that far.
void distance_pair(const Placed& a, const Placed& b, const coal::DistanceRequest& request, double upper_bound,
                   double cutoff, coal::GJKSolver& solver, coal::DistanceResult& result, CollisionRow* row) {
    result.clear();
    const Convex* solid_a = a.mesh->convex();
    const Convex* solid_b = b.mesh->convex();
    // coal measures from its first shape: a, unless a convex a goes second
    const bool swapped = swapped_for(a, b);
    double measured = 0.0;
    if (solid_a != nullptr && solid_b != nullptr) {
        start_solver(solver, request);
        solver.gjk_initial_guess = coal::GJKInitialGuess::BoundingVolumeGuess;  // as collide_pair
        solver.distance_upper_bound = cutoff;
        measured = coal::ShapeShapeDistance<coal::ConvexBase32, coal::ConvexBase32>(
            solid_a, a.transform, solid_b, b.transform, &solver, request, result);
        if (solver.gjk.status == coal::details::GJK::NoCollisionEarlyStopped) {
            write_far(row);  // proven farther than the bound
            return;
        }
    } else if (solid_a != nullptr || solid_b != nullptr) {
        const Placed& surface = swapped ? b : a;
        const Placed& solid = swapped ? a : b;
        start_solver(solver, request);
        result.min_distance = cutoff;
        MeshConvexDistance node;
        coal::initialize(node, surface.mesh->model(), surface.transform, solid.mesh->convex()->base(), solid.transform,
                         &solver, request, result);
        coal::distance(&node);
        measured = result.min_distance;
    } else {
        result.min_distance = cutoff;
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
        write_far(row);
        return;
    }
    write_witness(swapped, result.b1, result.b2, result.nearest_points, result.normal, measured, row);
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
    std::vector<std::array<std::int64_t, 2>> pairs;
    bool collide(coal::CollisionObject* a, coal::CollisionObject* b) override {
        // each object's user data is its instance's row, a real int64
        const std::int64_t i = *static_cast<const std::int64_t*>(a->getUserData());
        const std::int64_t j = *static_cast<const std::int64_t*>(b->getUserData());
        pairs.push_back({std::min(i, j), std::max(i, j)});
        return false;  // keep going: every candidate is wanted
    }
};

// Where each instance places its mesh. Each box bounds the mesh twice,
// carried into the scene by rotating each center and taking half extents
// through the absolute values of the rotated axes (Arvo): from the vertices'
// box, and from the hierarchy's root oriented box. Either bounds it, so
// their overlap does: the oriented box is far tighter for a mesh lying
// diagonally in its own frame, and looser for a cube, whose principal axes
// are arbitrary
std::vector<Placed> place(const std::vector<std::unique_ptr<CollisionMesh>>& meshes,
                          const trimkit::Instances& instances) {
    std::vector<Placed> placed(instances.count);
    for (std::size_t i = 0; i < instances.count; i++) {
        const double* m = instances.matrix(i);
        const CollisionMesh* mesh = meshes[instances.mesh(i)].get();
        coal::Matrix3s rotation;
        coal::Vec3s translation;
        for (Eigen::Index r = 0; r < 3; r++) {
            for (Eigen::Index c = 0; c < 3; c++) {
                rotation(r, c) = m[(4 * r) + c];
            }
            translation[r] = m[(4 * r) + 3];
        }
        // a mesh without faces never touches anything: a point where it is placed
        trimkit::PointBounds corners = {{translation[0], translation[1], translation[2]},
                                        {translation[0], translation[1], translation[2]}};
        if (!mesh->empty()) {
            // The vertices' box carried corner by corner, as tinybvh carries
            // a ray scene's: a box rebuilt from a center and half extent
            // loses its small end beside a far one ([-1, 1e17] would start
            // at 0), and an identity placement here is exact. Then narrowed
            // by the oriented root box, which coal holds as a center and half
            // extent, so it is padded by its own rounding first
            const trimkit::PointBounds& vertex = mesh->corners();
            const coal::OBB& oriented = mesh->bounds();
            const coal::Vec3s oriented_center = (rotation * oriented.To) + translation;
            const coal::Matrix3s oriented_axes = rotation * oriented.axes;
            for (Eigen::Index r = 0; r < 3; r++) {
                double low = translation[r];
                double high = translation[r];
                for (Eigen::Index c = 0; c < 3; c++) {
                    const double from = rotation(r, c) * vertex.lowest[c];
                    const double to = rotation(r, c) * vertex.highest[c];
                    low += std::min(from, to);
                    high += std::max(from, to);
                }
                const double oriented_half = oriented_axes.row(r).cwiseAbs().dot(oriented.extent);
                const double rounding = trimkit::BOX_SLACK * (std::abs(oriented_center[r]) + oriented_half);
                const auto axis = static_cast<std::size_t>(r);
                corners.lowest[axis] = std::max(oriented_center[r] - oriented_half - rounding, low);
                corners.highest[axis] = std::min(oriented_center[r] + oriented_half + rounding, high);
            }
            trimkit::refuse_placed_beyond(corners.lowest, corners.highest, i);
        }
        std::array<double, 6> box{};
        trimkit::center_half(corners.lowest, corners.highest, box.data());
        placed[i] = {mesh, coal::Transform3s(rotation, translation), box, corners};
    }
    return placed;
}

// the instance pairs a query names, the caller's copy checked: each in
// range before it is cast, so a 32-bit size_t never wraps one into it
void check_pairs(const std::vector<std::int64_t>& pairs, std::size_t count) {
    for (std::size_t p = 0; p < pairs.size() / 2; p++) {
        const std::int64_t a = pairs[2 * p];
        const std::int64_t b = pairs[(2 * p) + 1];
        if (!trimkit::indexes(a, count) || !trimkit::indexes(b, count)) {
            trimkit::refuse("pairs names an instance out of range at row " + std::to_string(p));
        }
        if (a == b) {
            trimkit::refuse("pairs names one instance twice at row " + std::to_string(p));
        }
    }
}

// Every pair of instances whose boxes may come within `margin`, as ascending
// (i < j) rows: coal's dynamic AABB tree, built for this query alone over
// the placed boxes (each grown by half the margin and by its slack), so
// nothing about the scene changes and threads never share it
std::vector<std::int64_t> overlapping(const std::vector<Placed>& placed, double margin) {
    std::vector<coal::CollisionObject> objects;
    objects.reserve(placed.size());
    // as many as there could be, written by index, then cut to those kept
    std::vector<coal::CollisionObject*> registered(placed.size());
    std::size_t kept = 0;
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
            const double half = box[3 + static_cast<std::size_t>(axis)] + (margin / 2.0) + slack(box);
            object.getAABB().min_[axis] = box[static_cast<std::size_t>(axis)] - half;
            object.getAABB().max_[axis] = box[static_cast<std::size_t>(axis)] + half;
        }
        registered[kept++] = &object;
    }
    registered.resize(kept);
    Candidates candidates;
    if (registered.size() > 1) {
        coal::DynamicAABBTreeArrayCollisionManager tree;
        tree.registerObjects(registered);
        tree.setup();
        tree.collide(&candidates);
    }
    // in ascending order, so answers never depend on the tree's shape.
    // Each is a candidate only: `may_touch` holds it to the margin, as it
    // holds the explicit pairs
    std::sort(candidates.pairs.begin(), candidates.pairs.end());
    std::vector<std::int64_t> rows(2 * candidates.pairs.size());
    for (std::size_t p = 0; p < candidates.pairs.size(); p++) {
        rows[2 * p] = candidates.pairs[p][0];
        rows[(2 * p) + 1] = candidates.pairs[p][1];
    }
    return rows;
}

}  // namespace

trimkit::CollisionMeshes::CollisionMeshes(const std::vector<CollisionMeshInput>& meshes) {
    meshes_.resize(meshes.size());
    for (std::size_t i = 0; i < meshes.size(); i++) {
        // a convex mesh is taken at its word: coal measures it as a solid,
        // walking its faces' edges to the vertex farthest along each
        // direction, so it needs faces, but whether they bound a convex
        // solid is the caller's promise to keep (an untrue one gives wrong
        // distances, never a crash)
        if (meshes[i].convex && meshes[i].face_count == 0) {
            refuse("meshes[" + std::to_string(i) + "] is convex, so it needs faces");
        }
        meshes_[i] = std::make_unique<CollisionMesh>(meshes[i]);
    }
}

trimkit::CollisionMeshes::~CollisionMeshes() = default;

bool trimkit::CollisionMeshes::convex(std::size_t mesh) const { return meshes_[mesh]->convex() != nullptr; }

void trimkit::CollisionMeshes::as_given(std::size_t mesh, std::vector<double>* vertices,
                                        std::vector<std::int64_t>* faces) const {
    meshes_[mesh]->as_given(vertices, faces);
}

std::vector<double> trimkit::CollisionMeshes::boxes(const Instances& instances) const {
    const std::vector<Placed> placed = place(meshes_, instances);
    std::vector<double> boxes(6 * placed.size());
    for (std::size_t i = 0; i < placed.size(); i++) {
        for (std::size_t axis = 0; axis < 3; axis++) {
            boxes[(6 * i) + axis] = placed[i].corners.lowest[axis];
            boxes[(6 * i) + 3 + axis] = placed[i].corners.highest[axis];
        }
    }
    return boxes;
}

void trimkit::CollisionMeshes::collides(const Instances& instances,
                                        const std::optional<std::vector<std::int64_t>>& pairs, double margin,
                                        std::int64_t maximum_contacts, const parallel::Stop& stop,
                                        std::vector<Unset<std::int64_t>>* touching,
                                        std::vector<Unset<CollisionRow>>* contacts) const {
    // NOLINTNEXTLINE(readability-simplify-boolean-expr): negated, so NaN is refused too
    if (!(margin >= 0.0 && margin <= COORDINATE_LIMIT)) {
        refuse(std::string("margin must be at least 0 and within ") + COORDINATE_LIMIT_TEXT);
    }
    if (maximum_contacts < 1) {
        refuse("maximum_contacts must be at least 1");
    }
    const std::vector<Placed> placed = place(meshes_, instances);
    // the pairs to test: the given rows, or the boxes' candidates
    std::vector<std::int64_t> candidates;
    if (pairs) {
        check_pairs(*pairs, placed.size());
    } else {
        candidates = overlapping(placed, margin);
    }
    const std::int64_t* rows = pairs ? pairs->data() : candidates.data();
    const std::size_t pair_count = (pairs ? pairs->size() : candidates.size()) / 2;
    if (contacts != nullptr) {
        // Every contact of each pair that touches, as many as
        // `maximum_contacts` per pair, as coal's contact request finds them
        // through the same cull and kernels: in pair order, each pair's in coal's
        coal::CollisionRequest request(coal::CONTACT, at_most(maximum_contacts));
        request.security_margin = margin;
        request.distance_upper_bound = 0;  // as below: separated pairs stop early
        Gathered<CollisionRow> found;
        gather_in_parallel<CollisionRow>(
            pair_count, &found,
            [&](std::size_t begin, std::size_t end, std::vector<CollisionRow>& kept) {
                coal::GJKSolver solver;
                coal::CollisionResult result;
                for (std::size_t p = begin; p < end; p++) {
                    const Placed& a = placed[static_cast<std::size_t>(rows[2 * p])];
                    const Placed& b = placed[static_cast<std::size_t>(rows[(2 * p) + 1])];
                    if (!may_touch(a, b, request) || !collide_pair(a, b, request, solver, result)) {
                        continue;
                    }
                    const bool swapped = swapped_for(a, b);
                    // coal signs two crossing triangles by how deep each
                    // reaches as a solid, but a surface has no inside: two of
                    // them are 0 apart once they touch, with no direction, as
                    // distance_pair measures them
                    const bool surfaces = a.mesh->convex() == nullptr && b.mesh->convex() == nullptr;
                    for (const coal::Contact& contact : result.getContacts()) {
                        CollisionRow& row = kept.emplace_back();
                        row.pair[0] = rows[2 * p];
                        row.pair[1] = rows[(2 * p) + 1];
                        const bool crossing = surfaces && !(contact.penetration_depth > 0.0);
                        const coal::Vec3s normal =
                            crossing ? coal::Vec3s::Constant(std::numeric_limits<double>::quiet_NaN()) : contact.normal;
                        write_witness(swapped, contact.b1, contact.b2, contact.nearest_points, normal,
                                      crossing ? 0.0 : contact.penetration_depth, &row);
                    }
                }
            },
            stop);
        *contacts = gathered_rows(found);
        return;
    }
    coal::CollisionRequest request(coal::NO_REQUEST, 1);
    request.security_margin = margin;
    // GJK stops at the first plane that separates a pair by more than the
    // margin (`set` raises this bound to it): coal's default bound is
    // infinite, which measured every separated pair's distance in full
    request.distance_upper_bound = 0;
    // one pass: each pair culled, then tested, keeping those that touch
    Gathered<std::int64_t> found;
    gather_in_parallel<std::int64_t>(
        pair_count, &found,
        [&](std::size_t begin, std::size_t end, std::vector<std::int64_t>& kept) {
            coal::GJKSolver solver;
            coal::CollisionResult result;
            for (std::size_t p = begin; p < end; p++) {
                const Placed& a = placed[static_cast<std::size_t>(rows[2 * p])];
                const Placed& b = placed[static_cast<std::size_t>(rows[(2 * p) + 1])];
                if (may_touch(a, b, request) && collide_pair(a, b, request, solver, result)) {
                    kept.push_back(rows[2 * p]);
                    kept.push_back(rows[(2 * p) + 1]);
                }
            }
        },
        stop);
    *touching = gathered_rows(found);
}

std::vector<trimkit::Unset<trimkit::CollisionRow>> trimkit::CollisionMeshes::distances(
    const Instances& instances, const std::optional<std::vector<std::int64_t>>& pairs, double upper_bound,
    const parallel::Stop& stop) const {
    if (!(upper_bound > 0.0)) {
        refuse("upper_bound must be positive");
    }
    const std::vector<Placed> placed = place(meshes_, instances);
    // every pair is measured, so every pair is named: the given rows, or
    // each i < j in numpy.triu_indices order, found from where each row
    // of them starts rather than listed
    const std::int64_t* rows = nullptr;
    std::vector<std::size_t> starts(placed.size());
    std::size_t pair_count = 0;
    if (pairs) {
        check_pairs(*pairs, placed.size());
        rows = pairs->data();
        pair_count = pairs->size() / 2;
    } else {
        // k(k - 1) / 2 of them: counted first, so the sum below can't wrap
        static_cast<void>(element_count(placed.size(), placed.empty() ? 0 : placed.size() - 1));
        for (std::size_t i = 0; i < placed.size(); i++) {
            starts[i] = pair_count;
            pair_count += placed.size() - 1 - i;
        }
    }
    // every row written once, by its pair, so none is filled first
    std::vector<Unset<CollisionRow>> measured(pair_count);
    CollisionRow* out = elements(measured);
    const coal::DistanceRequest request;
    // one pass: each pair culled, then measured
    in_parallel(
        pair_count,
        [&](std::size_t begin, std::size_t end) {
            coal::GJKSolver solver;
            coal::DistanceResult result;
            for (std::size_t p = begin; p < end; p++) {
                std::size_t first = 0;
                std::size_t second = 0;
                pair_at(rows, starts, p, &first, &second);
                out[p].pair[0] = static_cast<std::int64_t>(first);
                out[p].pair[1] = static_cast<std::int64_t>(second);
                const Placed& a = placed[first];
                const Placed& b = placed[second];
                const double cutoff = cutoff_for(a, b, upper_bound);
                if (may_be_within(a, b, cutoff)) {
                    distance_pair(a, b, request, upper_bound, cutoff, solver, result, &out[p]);
                } else {
                    write_far(&out[p]);
                }
            }
        },
        stop);
    return measured;
}
