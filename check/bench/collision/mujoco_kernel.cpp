// MuJoCo's collision and distance queries timed in C, with no Python between
// calls: what trimkit_robot.py's MuJoCo rows cost at their best. It reads what
// `trimkit_robot.py ROBOT MODE CONFIGS DIRECTORY` writes there: the margin-0
// and far-margin models, each configuration's geom poses, and the pairs.
//
// build against the mujoco-core wheel's library and headers:
//   M=$(python -c "import mujoco, os; print(os.path.dirname(mujoco.__file__))")
//   clang++ -O2 -std=c++17 mujoco_kernel.cpp -I$M/include $M/libmujoco.so.* -Wl,-rpath,$M -o mujoco_kernel
// usage: mujoco_kernel DIRECTORY

#include <mujoco/mujoco.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>
#include <thread>
#include <vector>

template <typename T>
std::vector<T> read_all(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    const std::vector<char> bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    std::vector<T> values(bytes.size() / sizeof(T));
    std::copy_n(bytes.data(), values.size() * sizeof(T), reinterpret_cast<char*>(values.data()));
    return values;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: mujoco_kernel DIRECTORY\n");
        return 2;
    }
    const std::string directory = argv[1];
    mjModel* touch = mj_loadModel((directory + "/touch.mjb").c_str(), nullptr);
    mjModel* far = mj_loadModel((directory + "/far.mjb").c_str(), nullptr);
    if (touch == nullptr || far == nullptr) {
        std::fprintf(stderr, "could not load the models in %s\n", directory.c_str());
        return 1;
    }
    mjData* touch_data = mj_makeData(touch);
    mjData* far_data = mj_makeData(far);
    // per configuration, per geom: geom_xpos (3) then geom_xmat (9)
    const std::vector<double> poses = read_all<double>(directory + "/poses.bin");
    const std::vector<std::int64_t> pairs = read_all<std::int64_t>(directory + "/pairs.bin");
    const double reach = read_all<double>(directory + "/reach.bin").at(0);
    const std::size_t geoms = static_cast<std::size_t>(touch->ngeom);
    const std::size_t configs = poses.size() / (12 * geoms);
    const auto place = [&](mjData* data, std::size_t c) {
        for (std::size_t g = 0; g < geoms; g++) {
            const double* pose = &poses[(c * geoms + g) * 12];
            std::copy_n(pose, 3, data->geom_xpos + 3 * g);
            std::copy_n(pose + 3, 9, data->geom_xmat + 9 * g);
        }
    };
    using clock = std::chrono::steady_clock;
    const auto per_config_us = [&](clock::duration spent) {
        return std::chrono::duration<double, std::micro>(spent).count() / static_cast<double>(configs);
    };
    // best of three passes over every configuration, each call timed alone
    double collide = 1e300, distance = 1e300, geom_distance = 1e300, sum = 0.0;
    for (int pass = 0; pass < 3; pass++) {
        clock::duration collide_time{}, distance_time{}, geom_time{};
        sum = 0.0;
        for (std::size_t c = 0; c < configs; c++) {
            place(touch_data, c);
            place(far_data, c);
            auto start = clock::now();
            mj_collision(touch, touch_data);
            collide_time += clock::now() - start;
            start = clock::now();
            mj_collision(far, far_data);
            mj_collision(touch, touch_data);
            distance_time += clock::now() - start;
            start = clock::now();
            for (std::size_t p = 0; p < pairs.size() / 2; p++) {
                mjtNum fromto[6];  // the witness points, as trimkit always returns them
                sum += mj_geomDistance(far, far_data, static_cast<int>(pairs[2 * p]),
                                       static_cast<int>(pairs[2 * p + 1]), reach, fromto);
            }
            geom_time += clock::now() - start;
        }
        collide = std::min(collide, per_config_us(collide_time));
        distance = std::min(distance, per_config_us(distance_time));
        geom_distance = std::min(geom_distance, per_config_us(geom_time));
    }
    std::printf(
        "mujoco in C, µs per configuration: collides %.1f, distances (two mj_collision) %.1f, "
        "distances (mj_geomDistance per pair) %.1f; checksum %.6f over %zu configurations\n",
        collide, distance, geom_distance, sum, configs);

    // MuJoCo's kernels as a vendored copy on trimkit's pool would run them:
    // each configuration's pairs split across `threads`, each thread with its
    // own mjData (one shared mjModel). A pair collides as mj_collision tests
    // an explicit pair: bounding spheres, then the narrowphase function its
    // table names (mjc_Convex for two meshes); a distance is mj_geomDistance.
    // Its best case: workers spin between configurations, so waking costs
    // nothing, and pairs are taken in chunks from one atomic counter. Every
    // thread count up to the machine's, as a pool would pick the best
    const std::size_t pair_count = pairs.size() / 2;
    for (const unsigned threads : {1u, 2u, 4u, 8u, 16u, 32u}) {
        if (threads > std::thread::hardware_concurrency()) {
            break;
        }
        std::vector<mjData*> datas(threads);
        for (mjData*& data : datas) {
            data = mj_makeData(far);
        }
        std::atomic<std::size_t> generation{0}, next{0}, finished{0};
        std::atomic<bool> quit{false};
        std::size_t config = 0;
        bool measure_distance = false;
        constexpr std::size_t LINE = 8;  // doubles or counts per cache line
        std::vector<double> sums(threads * LINE, 0.0);
        std::vector<std::uint64_t> hit_counts(threads * LINE, 0);
        constexpr std::size_t CHUNK = 8;
        const auto work = [&](unsigned t) {
            mjData* data = datas[t];
            place(data, config);
            mjPreContact contacts[mjMAXCONPAIR];
            mjtNum fromto[6];  // the witness points, as trimkit always returns them
            for (std::size_t begin; (begin = next.fetch_add(CHUNK)) < pair_count;) {
                for (std::size_t p = begin; p < std::min(begin + CHUNK, pair_count); p++) {
                    const int a = static_cast<int>(pairs[2 * p]), b = static_cast<int>(pairs[2 * p + 1]);
                    if (measure_distance) {
                        sums[t * LINE] += mj_geomDistance(far, data, a, b, reach, fromto);
                        continue;
                    }
                    const double* pa = data->geom_xpos + 3 * a;
                    const double* pb = data->geom_xpos + 3 * b;
                    const double bound = far->geom_rbound[a] + far->geom_rbound[b];
                    const double dx = pb[0] - pa[0], dy = pb[1] - pa[1], dz = pb[2] - pa[2];
                    if (dx * dx + dy * dy + dz * dz > bound * bound) {
                        continue;
                    }
                    const int lo = far->geom_type[a] <= far->geom_type[b] ? a : b;
                    const int hi = lo == a ? b : a;
                    hit_counts[t * LINE] +=
                        mjCOLLISIONFUNC[far->geom_type[lo]][far->geom_type[hi]](far, data, contacts, lo, hi, 0.0) > 0;
                }
            }
        };
        std::vector<std::thread> workers;
        for (unsigned t = 1; t < threads; t++) {
            workers.emplace_back([&, t] {
                for (std::size_t seen = 0;;) {
                    while (generation.load(std::memory_order_acquire) == seen) {
                        if (quit.load(std::memory_order_relaxed)) {
                            return;
                        }
                    }
                    seen = generation.load(std::memory_order_acquire);
                    if (quit.load(std::memory_order_relaxed)) {
                        return;  // woken to stop, not for a configuration
                    }
                    work(t);
                    finished.fetch_add(1, std::memory_order_acq_rel);
                }
            });
        }
        const auto run_all = [&](bool distance_pass) {
            measure_distance = distance_pass;
            const auto start = clock::now();
            for (config = 0; config < configs; config++) {
                next.store(0, std::memory_order_relaxed);
                finished.store(0, std::memory_order_relaxed);
                generation.fetch_add(1, std::memory_order_acq_rel);
                work(0);
                while (finished.load(std::memory_order_acquire) != threads - 1) {
                }
            }
            return per_config_us(clock::now() - start);
        };
        double threaded_collide = 1e300, threaded_distance = 1e300;
        for (int pass = 0; pass < 3; pass++) {
            std::fill(hit_counts.begin(), hit_counts.end(), 0);
            std::fill(sums.begin(), sums.end(), 0.0);
            threaded_collide = std::min(threaded_collide, run_all(false));
            threaded_distance = std::min(threaded_distance, run_all(true));
        }
        quit.store(true);
        generation.fetch_add(1);
        for (std::thread& worker : workers) {
            worker.join();
        }
        std::uint64_t hits = 0;
        double threaded_sum = 0.0;
        for (unsigned t = 0; t < threads; t++) {
            hits += hit_counts[t * LINE];
            threaded_sum += sums[t * LINE];
            mj_deleteData(datas[t]);
        }
        std::printf(
            "mujoco in C on %u spinning threads, µs per configuration: collides %.1f (%.2f hits per configuration), "
            "distances (mj_geomDistance) %.1f; checksum %.6f\n",
            threads, threaded_collide, static_cast<double>(hits) / static_cast<double>(configs), threaded_distance,
            threaded_sum);
    }
    mj_deleteData(touch_data);
    mj_deleteData(far_data);
    mj_deleteModel(touch);
    mj_deleteModel(far);
    return 0;
}
