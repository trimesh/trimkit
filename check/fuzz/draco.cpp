// Any bytes decode as `trimkit.exchange.draco_decode` does (src/draco.cpp),
// with no attributes asked for and with the first two: refused, or faces
// that index real points and attributes of exactly one tightly packed row a
// point, read only from values the buffer holds (ASan sees any other read).
// Draco asking for absurd amounts of memory is a decompression bomb, not a
// bug (a valid buffer of a few bytes can decode to millions of faces, and
// draco_decode has no size limit, as its docstring says), so `uv run scripts
// fuzz` runs this with out-of-memory reports ignored.
#include "../../src/trimkit.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    for (const std::vector<std::int64_t>& ids : {std::vector<std::int64_t>{}, std::vector<std::int64_t>{0, 1}}) {
        std::vector<std::int64_t> faces;
        std::vector<trimkit::DracoAttributeOutput> attributes;
        try {
            trimkit::draco_decode_mesh(data, size, ids, &faces, &attributes);
        } catch (const std::invalid_argument&) {
            continue;
        }
        if (faces.size() % 3 != 0 || attributes.size() != ids.size()) {
            __builtin_trap();
        }
        // every attribute one row a point, the same points for each
        std::size_t points = 0;
        for (std::size_t w = 0; w < attributes.size(); w++) {
            const trimkit::DracoAttributeOutput& a = attributes[w];
            const std::size_t row =
                a.components * trimkit::DRACO_COMPONENT_BYTES[static_cast<std::size_t>(a.component)];
            if (a.components == 0 || a.values.size() % row != 0 || (w > 0 && a.values.size() / row != points)) {
                __builtin_trap();
            }
            points = a.values.size() / row;
        }
        for (const std::int64_t index : faces) {
            if (index < 0 || (!attributes.empty() && static_cast<std::size_t>(index) >= points)) {
                __builtin_trap();
            }
        }
    }
    return 0;
}
