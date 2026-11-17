// ChunkList, which gathered rows are pushed onto from every thread of a
// call: a reader that finds a node sees what its pusher wrote into it (plain
// ints: a missing happens-before is a data race), while the pushes go on; and
// a take while they do, and one once they are done, between them find every
// node pushed exactly once
#include "harness.hpp"

struct Node {
    int payload = 0;
    Node* next = nullptr;
};

trimkit::parallel::ChunkList<Node> list;
Node nodes[2];
std::atomic<int> taken{0};
std::atomic<int> taken_sum{0};

void take_every() {
    for (Node* node = list.take_all(); node != nullptr; node = node->next) {
        taken.fetch_add(1, std::memory_order_relaxed);
        taken_sum.fetch_add(node->payload, std::memory_order_relaxed);
    }
}

int main() {
    auto push_one = [] {
        nodes[1].payload = 2;
        list.push(&nodes[1]);
    };
    const pthread_t thread = harness::start(push_one);
    for (Node* node = list.peek(); node != nullptr; node = node->next) {
        assert(node->payload == 2);  // only the other thread's can be there yet
    }
    nodes[0].payload = 1;
    list.push(&nodes[0]);
    auto take = [] { take_every(); };
    const pthread_t taker = harness::start(take);
    harness::join(thread);
    harness::join(taker);
    take_every();
    assert(taken.load(std::memory_order_relaxed) == 2 && taken_sum.load(std::memory_order_relaxed) == 3);
    return 0;
}
