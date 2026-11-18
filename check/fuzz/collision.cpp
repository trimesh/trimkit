// Property: for two small triangle meshes in any rigid poses, coal's
// bounding-volume traversal, called as src/collision.cpp calls it, agrees
// with testing every pair of triangles with the same leaf tests (its
// closed-form overlap for a query without contacts or margin, else GJK): it
// never reports a collision they don't, and misses none they find but a
// contact at zero distance, where bounding-volume rounding may fall either
// way; and it finds the same minimum distance. A bounding volume that
// didn't contain its triangles, or a pruning test that wasn't conservative,
// would differ.
#include <fuzzer/FuzzedDataProvider.h>

#include <coal/BVH/BVH_model.h>
#include <coal/collision_data.h>
#include <coal/internal/shape_shape_func.h>
#include <coal/internal/traversal_node_bvhs.h>
#include <coal/internal/traversal_node_setup.h>
#include <coal/narrowphase/narrowphase.h>
#include <collision_node.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

using Model = coal::BVHModel<coal::OBBRSS>;

struct Mesh {
    std::vector<coal::Vec3s> vertices;
    std::vector<coal::Triangle32> faces;
    Model model;
};

void read_mesh(FuzzedDataProvider& input, Mesh& mesh) {
    const auto vertex_count = input.ConsumeIntegralInRange<std::uint32_t>(3, 24);
    const auto face_count = input.ConsumeIntegralInRange<std::uint32_t>(1, 32);
    mesh.vertices.resize(vertex_count);
    for (coal::Vec3s& vertex : mesh.vertices) {
        for (Eigen::Index axis = 0; axis < 3; axis++) {
            vertex[axis] = input.ConsumeFloatingPointInRange<double>(-1.0, 1.0);
        }
    }
    mesh.faces.resize(face_count);
    for (coal::Triangle32& face : mesh.faces) {
        face = coal::Triangle32(input.ConsumeIntegralInRange<std::uint32_t>(0, vertex_count - 1),
                                input.ConsumeIntegralInRange<std::uint32_t>(0, vertex_count - 1),
                                input.ConsumeIntegralInRange<std::uint32_t>(0, vertex_count - 1));
    }
    mesh.model.beginModel(face_count, vertex_count);
    mesh.model.addSubModel(mesh.vertices, mesh.faces);
    mesh.model.endModel();
}

// a rigid transform from a fuzzed quaternion and translation
coal::Transform3s read_pose(FuzzedDataProvider& input) {
    double q[4];
    for (double& value : q) {
        value = input.ConsumeFloatingPointInRange<double>(-1.0, 1.0);
    }
    const double norm = std::sqrt((q[0] * q[0]) + (q[1] * q[1]) + (q[2] * q[2]) + (q[3] * q[3]));
    const coal::Quats rotation =
        norm < 1e-6 ? coal::Quats::Identity() : coal::Quats(q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm);
    coal::Vec3s translation;
    for (Eigen::Index axis = 0; axis < 3; axis++) {
        translation[axis] = input.ConsumeFloatingPointInRange<double>(-2.0, 2.0);
    }
    return coal::Transform3s(rotation.toRotationMatrix(), translation);
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    FuzzedDataProvider input(data, size);
    Mesh first;
    Mesh second;
    read_mesh(input, first);
    read_mesh(input, second);
    const coal::Transform3s first_pose = read_pose(input);
    const coal::Transform3s second_pose = read_pose(input);

    // the distance traversal, and the least distance over every pair of
    // triangles measured as its leaves measure them: in the first mesh's
    // frame, the second's carried there by the node's relative transform.
    // The exact triangle distance is ill-conditioned for triangles without
    // area, so any other frame can differ from the leaves' by ~1e-6
    const coal::DistanceRequest distance_request;
    coal::DistanceResult distance_result;
    coal::MeshDistanceTraversalNodeOBBRSS distance_node;
    coal::initialize(distance_node, first.model, first_pose, second.model, second_pose, distance_request,
                     distance_result);
    coal::distance(&distance_node);
    double least = std::numeric_limits<double>::infinity();
    double least_in_scene = std::numeric_limits<double>::infinity();
    // whether the closest pair has a triangle without area: see below
    bool least_degenerate = false;
    auto has_area = [](const coal::Vec3s& u, const coal::Vec3s& v, const coal::Vec3s& w) {
        return (v - u).cross(w - u).squaredNorm() > 0;
    };
    for (const coal::Triangle32& a : first.faces) {
        for (const coal::Triangle32& b : second.faces) {
            coal::Vec3s p;
            coal::Vec3s q;
            const coal::Vec3s s[3] = {first.vertices[a[0]], first.vertices[a[1]], first.vertices[a[2]]};
            coal::Vec3s t[3];
            for (std::size_t corner = 0; corner < 3; corner++) {
                t[corner] = (distance_node.RT._R() * second.vertices[b[corner]]) + distance_node.RT._T();
            }
            const double pair = std::sqrt(coal::TriangleDistance::sqrTriDistance(s, t, p, q));
            if (pair < least) {
                least = pair;
                least_degenerate = !has_area(s[0], s[1], s[2]) || !has_area(t[0], t[1], t[2]);
            }
            // and in the scene's, where the collision leaves' GJK measures
            const coal::Vec3s ws[3] = {first_pose.transform(s[0]), first_pose.transform(s[1]),
                                       first_pose.transform(s[2])};
            const coal::Vec3s wt[3] = {second_pose.transform(second.vertices[b[0]]),
                                       second_pose.transform(second.vertices[b[1]]),
                                       second_pose.transform(second.vertices[b[2]])};
            least_in_scene = std::min(least_in_scene, std::sqrt(coal::TriangleDistance::sqrTriDistance(ws, wt, p, q)));
        }
    }
    // the collision traversal measures in that frame too, but its oracle's
    // GJK runs in the scene's: rounding, ~1e-16 for coordinates near 1 in
    // squared distances, which a square root makes ~1e-8 near contact
    constexpr double ROUNDING = 1e-7;

    // collision, through the traversal and through every pair of triangles
    const coal::CollisionRequest collision_request(coal::NO_REQUEST, 1);
    coal::CollisionResult collision_result;
    coal::MeshCollisionTraversalNodeOBBRSS collision_node(collision_request);
    coal::initialize(collision_node, first.model, first_pose, second.model, second_pose, collision_result);
    coal::collide(&collision_node, collision_request, collision_result);
    // the leaf's own tests, on the triangles as the traversal places them:
    // the closed-form overlap for two triangles with area, on the second
    // mesh's carried into the first's frame by the node's relative transform
    // (so the arithmetic is the traversal's exactly), else GJK's distance
    coal::GJKSolver solver(collision_request);
    bool any_touch = false;
    // how deep the deepest touching pair interpenetrates, by EPA: a miss is
    // only allowed when every touching pair merely touches
    double deepest = 0.0;
    auto depth = [&](const coal::TriangleP& one, const coal::TriangleP& two) {
        coal::GJKSolver signed_solver;
        coal::Vec3s w1;
        coal::Vec3s w2;
        coal::Vec3s normal;
        return -coal::internal::ShapeShapeDistance<coal::TriangleP, coal::TriangleP>(
            &one, first_pose, &two, second_pose, &signed_solver, true, w1, w2, normal);
    };
    for (const coal::Triangle32& a : first.faces) {
        for (const coal::Triangle32& b : second.faces) {
            const coal::Vec3s& p0 = first.vertices[a[0]];
            const coal::Vec3s& p1 = first.vertices[a[1]];
            const coal::Vec3s& p2 = first.vertices[a[2]];
            const coal::Vec3s& q0 = second.vertices[b[0]];
            const coal::Vec3s& q1 = second.vertices[b[1]];
            const coal::Vec3s& q2 = second.vertices[b[2]];
            const bool both_have_area =
                (p1 - p0).cross(p2 - p0).squaredNorm() > 0 && (q1 - q0).cross(q2 - q0).squaredNorm() > 0;
            if (both_have_area) {
                coal::Vec3s q[3];
                for (std::size_t corner = 0; corner < 3; corner++) {
                    q[corner] = (collision_node.RT._R() * second.vertices[b[corner]]) + collision_node.RT._T();
                }
                coal::Scalar bound = 0;
                if (coal::internal::triangleTriangleOverlap(p0, p1, p2, q[0], q[1], q[2], bound)) {
                    any_touch = true;
                    deepest = std::max(deepest, depth(coal::TriangleP(p0, p1, p2), coal::TriangleP(q0, q1, q2)));
                }
                continue;
            }
            const coal::TriangleP one(p0, p1, p2);
            const coal::TriangleP two(q0, q1, q2);
            coal::Vec3s w1;
            coal::Vec3s w2;
            coal::Vec3s normal;
            const double distance = coal::internal::ShapeShapeDistance<coal::TriangleP, coal::TriangleP>(
                &one, first_pose, &two, second_pose, &solver, false, w1, w2, normal);
            if (distance <= collision_request.collision_distance_threshold) {
                any_touch = true;
                deepest = std::max(deepest, depth(one, two));
            }
        }
    }
    // the two may differ only where the meshes touch to within rounding, and
    // the traversal may never miss two triangles that truly interpenetrate
    const bool touching = std::min(least, least_in_scene) <= ROUNDING;
    if (collision_result.isCollision() != any_touch && (!touching || deepest > ROUNDING)) {
        __builtin_trap();
    }

    // The traversal measures each leaf exactly as above, so it differs only
    // by pruning a closer pair. It does, by up to ~1e-5, when the closest pair
    // has a triangle without area (a segment or a point): 5.5e-6 for two
    // such triangles touching, the same on coal devel without our patch
    // (vendor/patches/coal-bvh-bvh-performance.patch). Collision is unaffected
    constexpr double DEGENERATE = 1e-5;
    if (std::abs(distance_result.min_distance - least) > (least_degenerate ? DEGENERATE : ROUNDING)) {
        __builtin_trap();
    }

    return 0;
}
