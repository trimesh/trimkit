// svgpath: parse SVG path data (the `d` attribute) into vertices and one
// record per segment, and write them back as path data.
//
// Header-only, written as plain C-style C++17: structs, functions and loops,
// depending only on the standard library, fast_float and Dragonbox. A path
// is two arrays in absolute coordinates:
//
// - vertices: x, y pairs. Every point a segment names, once, in path order:
//   on-curve points, and Bezier and conic control points. Never a center.
// - segments: one SvgSegment each, naming its points by vertex index in four
//   slots (start, a control or an arc's halfway point, a cubic's second
//   control, end; -1 where a kind has none), with what else its kind needs
//   in `kind_data`: an arc's center, or a conic's weight.
//
// Connectivity and closure are index equality. A segment that doesn't start
// on the vertex the previous one ended on starts a new subpath. A line that
// ends on its subpath's first vertex, without starting there, is the `Z`
// that closes it. Relative commands are resolved, H and V become lines, and
// S and T reflections become explicit control points.
//
// Arcs are split into pieces of at most MAX_ARC (120 degrees). An arc whose
// radii are equal becomes ARC pieces: three points on a circle. Any other
// becomes CONIC pieces: an exact rational quadratic Bezier, which affine
// transforms carry exactly. `format` writes ARC and CONIC pieces as `A`
// commands, and splits any piece longer than MAX_ARC, so every piece it
// writes is at most 120 degrees and never takes the large-arc flag.
//
// `format` is the inverse of `parse`. At 17 digits, parsing what it writes
// reproduces every vertex of a line or Bezier, and every segment's start and
// end, bit for bit. An arc's halfway point and center, or a conic's control
// and weight, come back within rounding.
//
// Every function is `static inline`: each file that includes this header
// gets its own private copy, so two different versions can never collide.
//
// Copyright (c) 2026 Michael Dawson-Haggerty, MIT License.
#pragma once

#include <dragonbox.h>
#include <fast_float/fast_float.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace svgpath {

// the values of `SvgSegment::kind`: never renumbered, only added to
enum SvgKind : std::uint8_t { LINE = 0, QUADRATIC = 1, CUBIC = 2, ARC = 3, CONIC = 4 };

// One segment, 56 bytes, viewed by numpy as a structured array.
struct SvgSegment {
    std::uint8_t kind;
    std::uint8_t padding[7];  // zero
    // vertex indices: start, then a curve's control (QUADRATIC, CONIC), a
    // cubic's first control or an arc's halfway point, then a cubic's
    // second control, then end; -1 where the kind has none
    std::int64_t points[4];
    // what the kind needs beyond its points: an ARC's center, x and y, as
    // `parse` computed it, the same for every piece of one arc, which
    // `format` never reads; a CONIC's weight, the cosine of half its sweep,
    // then 0; zeros for any other kind
    double kind_data[2];
};
static_assert(sizeof(SvgSegment) == 56, "numpy views SvgSegment as 56 bytes");

// A parsed path: `vertices` holds x, y pairs.
struct SvgPath {
    std::vector<double> vertices;
    std::vector<SvgSegment> segments;
};

static const double PI = 3.14159265358979323846;
// no coordinate, or number that makes one, beyond +-1e30: the bound trimkit
// holds every float64 input to (src/numpy.hpp), so what `parse` produces is
// exactly what `format` and everything downstream accept
static constexpr double COORDINATE_LIMIT = 1e30;
// MAX_ARC is 120 degrees, held as the cosine of half of it. A conic's weight
// is the cosine of half its sweep, so this is also the smallest weight.
static constexpr double MAX_ARC_HALF_COSINE = 0.5;
// How far past MAX_ARC, in cosines, `format` lets a piece run before
// splitting it. A piece `parse` made at most MAX_ARC, recomputed from its
// rounded points, can land just past it, and shouldn't be split again.
static constexpr double SPLIT_SLACK = 1e-6;

static inline bool within(double value) {
    return std::abs(value) <= COORDINATE_LIMIT;  // false for NaN, too
}

// Whether a radius `format` derives is one it can write, which `parse`
// reads back. One within rounding of the bound is taken as the bound: an
// arc `parse` made with a radius of exactly 1e30 comes back an ulp or so
// past it, and would otherwise read back as a line.
static inline bool writable_radius(double* radius) {
    if (*radius > COORDINATE_LIMIT && *radius <= COORDINATE_LIMIT * (1.0 + 1e-12)) {
        *radius = COORDINATE_LIMIT;
    }
    return *radius >= std::numeric_limits<double>::min() && *radius <= COORDINATE_LIMIT;
}

// how many numbers a command takes, or -1 for a letter that isn't one
static inline int argument_count(char upper) {
    switch (upper) {
        case 'Z':
            return 0;
        case 'H':
        case 'V':
            return 1;
        case 'M':
        case 'L':
        case 'T':
            return 2;
        case 'S':
        case 'Q':
            return 4;
        case 'C':
            return 6;
        case 'A':
            return 7;
        default:
            return -1;
    }
}

// the most numbers any command takes: an arc's seven
static const int MOST_ARGUMENTS = 7;

static inline bool is_separator(char character) {
    return character == ' ' || character == ',' || character == '\t' || character == '\n' || character == '\r' ||
           character == '\f';
}

// whether a number can start with this character, which is also how an
// implicitly repeated command is recognized
static inline bool is_number_start(char character) {
    return (character >= '0' && character <= '9') || character == '.' || character == '-' || character == '+';
}

static inline bool is_letter(char character) {
    return (character >= 'a' && character <= 'z') || (character >= 'A' && character <= 'Z');
}

// Path data being read. Every access is bounds checked: `peek` past the end
// is '\0', which is no kind of character, and fast_float is given the end.
struct PathDataReader {
    const char* text;
    std::size_t size;
    std::size_t offset;
};

static inline char peek(const PathDataReader* reader) {
    return reader->offset < reader->size ? reader->text[reader->offset] : '\0';
}

// skip whitespace and commas, and return whether anything is left
static inline bool skip_separators(PathDataReader* reader) {
    while (reader->offset < reader->size && is_separator(reader->text[reader->offset])) {
        reader->offset++;
    }
    return reader->offset < reader->size;
}

static inline bool read_number(PathDataReader* reader, double* value) {
    if (!skip_separators(reader) || !is_number_start(peek(reader))) {
        return false;
    }
    const fast_float::chars_format grammar = fast_float::chars_format::general | fast_float::chars_format::no_infnan |
                                             fast_float::chars_format::allow_leading_plus;
    const char* start = reader->text + reader->offset;
    const char* end = reader->text + reader->size;
    const fast_float::from_chars_result result = fast_float::from_chars(start, end, *value, grammar);
    // out of range is also how underflow is reported, and a number too small
    // for a double is still a number: only overflow, or beyond the coordinate
    // bound, is an error
    const bool parsed = result.ec == std::errc() || result.ec == std::errc::result_out_of_range;
    if (!parsed || !within(*value)) {
        return false;
    }
    reader->offset += static_cast<std::size_t>(result.ptr - start);
    return true;
}

// an arc flag is a single character, so "001" is three flags
static inline bool read_flag(PathDataReader* reader, double* value) {
    if (!skip_separators(reader) || (peek(reader) != '0' && peek(reader) != '1')) {
        return false;
    }
    *value = peek(reader) == '1' ? 1.0 : 0.0;
    reader->offset++;
    return true;
}

// An ARC or CONIC as `format` writes it: `count` `A` commands on one
// ellipse, each at most MAX_ARC and so never the large arc, the last ending
// on the segment's end.
struct SvgArcCommands {
    double radius_x;
    double radius_y;
    // twice the rotation, as a direction: only `format` needs the angle,
    // so it alone takes the arctangent
    double doubled_rotation[2];
    double sweep;  // the sweep flag: 1 when the angle increases along it
    int count;     // 1, 2 or 4
    double ends[3][2];
};

// Why an ARC (start, a point on it, end: three points on a circle) or a
// CONIC (start, control, end and its weight) can't be written as path data,
// or nullptr with its `A` commands in `commands`: the one definition of a
// writable arc. `format` refuses by it, and `parse` makes a line of any arc
// it refuses, so whatever `parse` produces `format` writes. The points are
// already finite and within the coordinate bound.
static inline const char* arc_commands(std::uint8_t kind, const double* start, const double* through, const double* end,
                                       double weight, SvgArcCommands* commands) {
    // Every product below is of at most four differences of coordinates.
    // Within the bound none can overflow (2e30 to the fourth is 1.6e121),
    // but tiny ones could underflow, so values whose largest is below 2^-200
    // are first scaled by a power of two: exact, and undone on the result
    const auto scale_up = [](double* values) {
        double largest = 0.0;
        for (int index = 0; index < 4; index++) {
            largest = std::fmax(largest, std::abs(values[index]));
        }
        int exponent = 0;
        if (largest < 0x1p-200) {
            static_cast<void>(std::frexp(largest, &exponent));
            for (int index = 0; index < 4; index++) {
                values[index] = std::ldexp(values[index], -exponent);
            }
        }
        return exponent;
    };
    // a and b: the start and end relative to the middle point
    double relative[4] = {start[0] - through[0], start[1] - through[1], end[0] - through[0], end[1] - through[1]};
    const int exponent = scale_up(relative);
    const double a_x = relative[0];
    const double a_y = relative[1];
    const double b_x = relative[2];
    const double b_y = relative[3];
    // a x b: negative when start, middle, end turn counterclockwise (the
    // angle increasing, the sweep flag 1), zero when they are on one line
    const double cross = (a_x * b_y) - (a_y * b_x);
    if (!(cross != 0.0)) {
        return "an arc's or conic's three points must not lie on one line";
    }
    commands->sweep = cross < 0.0 ? 1.0 : 0.0;
    const double turn = cross < 0.0 ? 1.0 : -1.0;
    commands->count = 1;

    if (kind == ARC) {
        // The circle through all three: radius |a||b||a - b| / 2|a x b|. The
        // angle at the middle point is 180 degrees less half the sweep, so
        // a sweep past MAX_ARC is an angle there with a cosine past -1/2,
        // and past twice MAX_ARC one past +1/2
        const double square_a = (a_x * a_x) + (a_y * a_y);
        const double square_b = (b_x * b_x) + (b_y * b_y);
        const double square_chord = ((b_x - a_x) * (b_x - a_x)) + ((b_y - a_y) * (b_y - a_y));
        double radius = std::ldexp(std::sqrt(square_a * square_b * square_chord) / (2.0 * std::abs(cross)), exponent);
        if (!writable_radius(&radius)) {
            return "an arc's circle must have a radius that is normal and within \xc2\xb1"
                   "1e30";
        }
        commands->radius_x = radius;
        commands->radius_y = radius;
        commands->doubled_rotation[0] = 1.0;
        commands->doubled_rotation[1] = 0.0;
        const double cosine = ((a_x * b_x) + (a_y * b_y)) / std::sqrt(square_a * square_b);
        if (cosine > -MAX_ARC_HALF_COSINE + SPLIT_SLACK) {
            commands->count = cosine > MAX_ARC_HALF_COSINE + SPLIT_SLACK ? 4 : 2;
            // the center, then each boundary as the point halfway along the
            // arc between two others: their difference turned a quarter turn
            const double denominator = 2.0 * cross;
            const double center_x =
                through[0] + std::ldexp(((b_y * square_a) - (a_y * square_b)) / denominator, exponent);
            const double center_y =
                through[1] + std::ldexp(((a_x * square_b) - (b_x * square_a)) / denominator, exponent);
            const auto halfway = [&](const double* from, const double* to, double* point) {
                const double difference_x = to[0] - from[0];
                const double difference_y = to[1] - from[1];
                const double length = std::hypot(difference_x, difference_y);
                point[0] = center_x + (radius * turn * difference_y / length);
                point[1] = center_y - (radius * turn * difference_x / length);
            };
            if (commands->count == 2) {
                halfway(start, end, commands->ends[0]);
            } else {
                halfway(start, end, commands->ends[1]);
                halfway(start, commands->ends[1], commands->ends[0]);
                halfway(commands->ends[1], end, commands->ends[2]);
            }
        }
    } else {
        // NOLINTNEXTLINE(readability-simplify-boolean-expr): negated, so NaN is refused too
        if (!(weight > 0.0 && weight < 1.0)) {
            return "a conic's weight must be between 0 and 1";
        }
        // The ellipse of a rational quadratic with weight w = cos(half its
        // sweep) has two conjugate semi-diameters: f, w / (1 - w^2) times the
        // control less the chord's midpoint, and g, the half chord over
        // sqrt(1 - w^2). The radii are the square roots of the eigenvalues of
        // [f g][f g]^T = [[p, q], [q, r]], the smaller as the determinant of
        // [f g] over the larger, which doesn't cancel, and the rotation is
        // the larger's eigenvector, at half the angle of (p - r, 2q)
        const double complement = (1.0 - weight) * (1.0 + weight);
        const double along = weight / complement;
        const double across = 0.5 / std::sqrt(complement);
        double axes[4] = {along * (through[0] - (0.5 * (start[0] + end[0]))),
                          along * (through[1] - (0.5 * (start[1] + end[1]))), across * (end[0] - start[0]),
                          across * (end[1] - start[1])};
        const int size = scale_up(axes);
        const double p = (axes[0] * axes[0]) + (axes[2] * axes[2]);
        const double r = (axes[1] * axes[1]) + (axes[3] * axes[3]);
        const double q = (axes[0] * axes[1]) + (axes[2] * axes[3]);
        const double half_difference = (p - r) / 2.0;
        const double larger = std::sqrt(((p + r) / 2.0) + std::sqrt((half_difference * half_difference) + (q * q)));
        commands->radius_x = std::ldexp(larger, size);
        commands->radius_y = std::ldexp(std::abs((axes[0] * axes[3]) - (axes[2] * axes[1])) / larger, size);
        commands->doubled_rotation[0] = half_difference;
        commands->doubled_rotation[1] = q;
        if (!writable_radius(&commands->radius_x) || !writable_radius(&commands->radius_y)) {
            return "a conic's ellipse must have radii that are normal and within \xc2\xb1"
                   "1e30";
        }
        // a conic past MAX_ARC is split at its middle, (S + 2wC + E) / (2 + 2w)
        if (weight < MAX_ARC_HALF_COSINE - SPLIT_SLACK) {
            commands->count = 2;
            for (int axis = 0; axis < 2; axis++) {
                commands->ends[0][axis] =
                    (start[axis] + (2.0 * weight * through[axis]) + end[axis]) / (2.0 + (2.0 * weight));
            }
        }
    }
    for (int index = 0; index + 1 < commands->count; index++) {
        if (!within(commands->ends[index][0]) || !within(commands->ends[index][1])) {
            return "an arc split at MAX_ARC must have every piece's end within \xc2\xb1"
                   "1e30";
        }
    }
    return nullptr;
}

// Where a parse is: the current point and its vertex, and the subpath's
// start, where `Z` returns, and its vertex. A vertex is -1 until a segment
// needs it, so a moveto that draws nothing adds nothing.
struct SvgPen {
    SvgPath* out;
    double x;
    double y;
    std::int64_t vertex;
    double start_x;
    double start_y;
    std::int64_t start_vertex;
    // whether a vertex was added beyond the coordinate bound, which `format`
    // would refuse: that command is malformed, and ends parsing
    bool beyond;
};

static inline std::int64_t add_vertex(SvgPen* pen, double x, double y) {
    pen->beyond = pen->beyond || !within(x) || !within(y);
    pen->out->vertices.push_back(x);
    pen->out->vertices.push_back(y);
    return static_cast<std::int64_t>(pen->out->vertices.size() / 2) - 1;
}

// the vertex the pen is on, added the first time a segment needs it
static inline std::int64_t pen_vertex(SvgPen* pen) {
    if (pen->vertex < 0) {
        pen->vertex = add_vertex(pen, pen->x, pen->y);
        if (pen->start_vertex < 0) {
            pen->start_vertex = pen->vertex;  // only ever right after a moveto
        }
    }
    return pen->vertex;
}

// a segment from the pen, whose vertex its start already is, and the pen
// moved to its end
static inline void add_segment(SvgPen* pen, const SvgSegment* segment, double end_x, double end_y) {
    pen->out->segments.push_back(*segment);
    pen->vertex = segment->points[3];
    pen->x = end_x;
    pen->y = end_y;
}

// a straight line from the pen to a new vertex
static inline void add_line(SvgPen* pen, double end_x, double end_y) {
    const std::int64_t start = pen_vertex(pen);
    const SvgSegment segment = {LINE, {}, {start, -1, -1, add_vertex(pen, end_x, end_y)}, {0.0, 0.0}};
    add_segment(pen, &segment, end_x, end_y);
}

// An SVG arc from the pen, F.6.5, as one, two or four pieces of at most
// MAX_ARC: ARC pieces for a circle, CONIC pieces for any other ellipse.
//
// It works in the unit frame, where the ellipse is the unit circle: unrotated,
// each axis divided by its radius, with the chord's midpoint at the origin.
// There the start is `ratio` and the end `-ratio`, and the center sits at
// -h times the unit normal from the chord toward the arc's middle, where h is
// the cosine of half the sweep: positive for the small arc, negative for the
// large. Every point is mapped back as the chord's midpoint plus its offset
// rotated and scaled by the radii, so a huge radius doesn't lose a short
// arc's middle to cancellation against its far-off center. No angle is ever
// computed, and a circle needs no trigonometry at all.
static inline void add_arc(SvgPen* pen, double end_x, double end_y, double radius_x, double radius_y, double rotation,
                           bool large_arc, bool sweep_positive) {
    if (end_x == pen->x && end_y == pen->y) {
        return;  // an arc that ends where it starts draws nothing, F.6.2
    }
    // a zero radius is a straight line, F.6.2, and so is any arc that can't
    // be described in normal doubles within the coordinate bound
    if (radius_x == 0.0 || radius_y == 0.0) {
        add_line(pen, end_x, end_y);
        return;
    }
    // a circle's rotation changes nothing, so it is taken as zero
    const bool circle = radius_x == radius_y;
    const double angle = circle ? 0.0 : rotation * PI / 180.0;
    const double cos_angle = circle ? 1.0 : std::cos(angle);
    const double sin_angle = circle ? 0.0 : std::sin(angle);
    const double half_x = (pen->x - end_x) / 2.0;
    const double half_y = (pen->y - end_y) / 2.0;
    double ratio_x = ((cos_angle * half_x) + (sin_angle * half_y)) / radius_x;
    double ratio_y = ((-sin_angle * half_x) + (cos_angle * half_y)) / radius_y;
    double reach = std::hypot(ratio_x, ratio_y);
    // a chord below the radii's precision (the ratios underflowed) is a line
    if (reach == 0.0) {
        add_line(pen, end_x, end_y);
        return;
    }
    if (!(reach <= 1.0)) {
        // radii too small to reach both endpoints are scaled up together,
        // F.6.6, which leaves half the ellipse: past the coordinate bound, or
        // subnormal (too imprecise to write back), they make a line
        radius_x *= reach;
        radius_y *= reach;
        // NOLINTNEXTLINE(readability-simplify-boolean-expr): negated, so NaN is refused too
        if (!(radius_x >= std::numeric_limits<double>::min() && radius_y >= std::numeric_limits<double>::min() &&
              radius_x <= COORDINATE_LIMIT && radius_y <= COORDINATE_LIMIT)) {
            add_line(pen, end_x, end_y);
            return;
        }
        ratio_x /= reach;
        ratio_y /= reach;
        reach = 1.0;
    }
    const double root = std::sqrt((1.0 - reach) * (1.0 + reach));
    const double h = large_arc ? -root : root;
    const double turn = sweep_positive ? 1.0 : -1.0;
    const double normal_x = turn * -ratio_y / reach;
    const double normal_y = turn * ratio_x / reach;
    const double center_x = -h * normal_x;
    const double center_y = -h * normal_y;
    const double chord_x = (pen->x + end_x) / 2.0;
    const double chord_y = (pen->y + end_y) / 2.0;
    const auto to_world = [&](double unit_x, double unit_y, double* point) {
        const double x = radius_x * unit_x;
        const double y = radius_y * unit_y;
        point[0] = chord_x + (cos_angle * x) - (sin_angle * y);
        point[1] = chord_y + (sin_angle * x) + (cos_angle * y);
    };
    // a circle's center, the same for every piece, and within the bound,
    // or the arc is a line
    double center[2] = {0.0, 0.0};
    if (circle) {
        to_world(center_x, center_y, center);
        if (!within(center[0]) || !within(center[1])) {
            add_line(pen, end_x, end_y);
            return;
        }
    }

    SvgKind kind = circle ? ARC : CONIC;
    int count = 1;
    double through[4][2];  // each piece's halfway point (ARC) or control (CONIC)
    double ends[4][2];     // each piece's end: the last is the arc's own
    double weights[4] = {};
    if (h >= MAX_ARC_HALF_COSINE) {
        // One piece. Its middle is (1 - h) along the normal and its control
        // (1/h - h): both computed as reach^2 over something, as 1 - h
        // cancels for a short arc
        const double offset = circle ? reach * reach / (1.0 + h) : reach * reach / h;
        to_world(offset * normal_x, offset * normal_y, through[0]);
        weights[0] = h;
        // a weight of exactly 1 is a quadratic Bezier, and needs no weight
        if (!circle && h == 1.0) {
            kind = QUADRATIC;
        }
    } else {
        // Two pieces past MAX_ARC, four past twice it: each boundary, and
        // each ARC piece's halfway point, is halfway along the arc between
        // two unit vectors from the center, their difference turned a
        // quarter turn in the direction of travel
        count = h >= -MAX_ARC_HALF_COSINE ? 2 : 4;
        const auto halfway = [turn](const double* from, const double* to, double* unit) {
            const double difference_x = to[0] - from[0];
            const double difference_y = to[1] - from[1];
            // unit vectors at least 60 degrees apart: no overflow or underflow
            const double length = std::sqrt((difference_x * difference_x) + (difference_y * difference_y));
            unit[0] = turn * difference_y / length;
            unit[1] = -turn * difference_x / length;
        };
        double bounds[5][2];
        bounds[0][0] = ratio_x - center_x;
        bounds[0][1] = ratio_y - center_y;
        bounds[count][0] = -ratio_x - center_x;
        bounds[count][1] = -ratio_y - center_y;
        bounds[count / 2][0] = normal_x;
        bounds[count / 2][1] = normal_y;
        if (count == 4) {
            halfway(bounds[0], bounds[2], bounds[1]);
            halfway(bounds[2], bounds[4], bounds[3]);
        }
        for (int piece = 0; piece < count; piece++) {
            const double* from = bounds[piece];
            const double* to = bounds[piece + 1];
            double unit[2];
            if (circle) {
                halfway(from, to, unit);
            } else {
                // the tangents' meeting point, (a + b) / (1 + a.b), and the
                // weight, sqrt((1 + a.b) / 2)
                const double dot = 1.0 + (from[0] * to[0]) + (from[1] * to[1]);
                unit[0] = (from[0] + to[0]) / dot;
                unit[1] = (from[1] + to[1]) / dot;
                weights[piece] = std::sqrt(dot / 2.0);
            }
            to_world(center_x + unit[0], center_y + unit[1], through[piece]);
            to_world(center_x + to[0], center_y + to[1], ends[piece]);
        }
    }
    ends[count - 1][0] = end_x;
    ends[count - 1][1] = end_y;

    // every piece must be one `format` writes, or the whole arc is a line
    double from[2] = {pen->x, pen->y};
    for (int piece = 0; piece < count; piece++) {
        SvgArcCommands commands{};
        const bool inside =
            within(through[piece][0]) && within(through[piece][1]) && within(ends[piece][0]) && within(ends[piece][1]);
        if (!inside || (kind != QUADRATIC &&
                        arc_commands(kind, from, through[piece], ends[piece], weights[piece], &commands) != nullptr)) {
            add_line(pen, end_x, end_y);
            return;
        }
        from[0] = ends[piece][0];
        from[1] = ends[piece][1];
    }
    for (int piece = 0; piece < count; piece++) {
        const std::int64_t start = pen_vertex(pen);
        const std::int64_t middle = add_vertex(pen, through[piece][0], through[piece][1]);
        SvgSegment segment = {kind, {}, {start, middle, -1, add_vertex(pen, ends[piece][0], ends[piece][1])}, {}};
        if (kind == ARC) {
            segment.kind_data[0] = center[0];
            segment.kind_data[1] = center[1];
        } else if (kind == CONIC) {
            segment.kind_data[0] = weights[piece];
        }
        add_segment(pen, &segment, ends[piece][0], ends[piece][1]);
    }
}

// Parse path data, appending to `out`. Returns how many bytes were parsed:
// all of them on success, else the offset of the first malformed token, with
// `out` holding everything before it, as an SVG renderer would draw.
//
// What it produces is linear in `size`: each segment takes at least two bytes
// of path data, or one for a `Z` after another segment, and an arc of at
// least twelve makes at most four pieces. The most is a line and its close
// in three bytes, two segments and a vertex: at most 43 bytes of output per
// byte of input, on top of what the vectors' growth leaves unused.
static inline std::size_t parse(const char* text, std::size_t size, SvgPath* out) {
    PathDataReader reader = {text, size, 0};
    SvgPen pen = {out, 0.0, 0.0, -1, 0.0, 0.0, -1, false};
    // room for typical path data, about five bytes a line, so it rarely grows
    out->segments.reserve(out->segments.size() + (size / 4));
    out->vertices.reserve(out->vertices.size() + (size / 2));
    double reflect_x = 0.0;  // the last control point, which S and T reflect
    double reflect_y = 0.0;
    char previous = 0;   // the previous command, upper case
    char command = 0;    // the current command, as written
    bool moved = false;  // path data must start with a moveto

    while (skip_separators(&reader)) {
        const std::size_t token = reader.offset;
        if (is_letter(peek(&reader))) {
            command = peek(&reader);
            reader.offset++;
        } else if (command == 0 || command == 'Z' || command == 'z' || !is_number_start(peek(&reader))) {
            // numbers need a command to repeat, and `Z` takes none
            return token;
        }
        const bool relative = command >= 'a';
        const char upper = relative ? static_cast<char>(command - 'a' + 'A') : command;
        const int count = argument_count(upper);
        if (count < 0 || (!moved && upper != 'M')) {
            return token;  // not a command, or path data not starting with a moveto
        }

        // every number, or stop at the first that is malformed or missing:
        // an arc's fourth and fifth are its single-character flags
        double values[MOST_ARGUMENTS] = {};
        for (int index = 0; index < count; index++) {
            const bool flag = upper == 'A' && (index == 3 || index == 4);
            const bool read = flag ? read_flag(&reader, &values[index]) : read_number(&reader, &values[index]);
            if (!read) {
                return reader.offset;
            }
        }
        const double origin_x = relative ? pen.x : 0.0;
        const double origin_y = relative ? pen.y : 0.0;
        const std::size_t vertices_before = out->vertices.size();
        const std::size_t segments_before = out->segments.size();
        const std::int64_t pen_vertex_before = pen.vertex;

        switch (upper) {
            case 'M':
                pen.x = origin_x + values[0];
                pen.y = origin_y + values[1];
                pen.start_x = pen.x;
                pen.start_y = pen.y;
                pen.vertex = -1;
                pen.start_vertex = -1;
                moved = true;
                // any further coordinate pairs are implicit linetos
                command = relative ? 'l' : 'L';
                break;
            case 'Z': {
                // the pen is on the first vertex only before anything is
                // drawn, after a moveto or a close: then there's nothing to close
                if (pen.vertex < 0 || pen.vertex == pen.start_vertex) {
                    break;
                }
                // back to the subpath's first vertex, shared rather than repeated
                const SvgSegment segment = {LINE, {}, {pen.vertex, -1, -1, pen.start_vertex}, {0.0, 0.0}};
                add_segment(&pen, &segment, pen.start_x, pen.start_y);
                break;
            }
            case 'L':
                add_line(&pen, origin_x + values[0], origin_y + values[1]);
                break;
            case 'H':
                add_line(&pen, origin_x + values[0], pen.y);
                break;
            case 'V':
                add_line(&pen, pen.x, origin_y + values[0]);
                break;
            case 'C':
            case 'S': {
                // S starts from the last control point reflected about the
                // pen when it follows a cubic, else from the pen
                double control_x = pen.x;
                double control_y = pen.y;
                if (upper == 'C') {
                    control_x = origin_x + values[0];
                    control_y = origin_y + values[1];
                } else if (previous == 'C' || previous == 'S') {
                    control_x = 2.0 * pen.x - reflect_x;
                    control_y = 2.0 * pen.y - reflect_y;
                }
                reflect_x = origin_x + values[count - 4];
                reflect_y = origin_y + values[count - 3];
                const double end_x = origin_x + values[count - 2];
                const double end_y = origin_y + values[count - 1];
                const std::int64_t start = pen_vertex(&pen);
                const std::int64_t first = add_vertex(&pen, control_x, control_y);
                const std::int64_t second = add_vertex(&pen, reflect_x, reflect_y);
                const SvgSegment segment = {CUBIC, {}, {start, first, second, add_vertex(&pen, end_x, end_y)}, {}};
                add_segment(&pen, &segment, end_x, end_y);
                break;
            }
            case 'Q':
            case 'T': {
                // T reflects the last control point when it follows a quadratic
                if (upper == 'Q') {
                    reflect_x = origin_x + values[0];
                    reflect_y = origin_y + values[1];
                } else if (previous == 'Q' || previous == 'T') {
                    reflect_x = 2.0 * pen.x - reflect_x;
                    reflect_y = 2.0 * pen.y - reflect_y;
                } else {
                    reflect_x = pen.x;
                    reflect_y = pen.y;
                }
                const double end_x = origin_x + values[count - 2];
                const double end_y = origin_y + values[count - 1];
                const std::int64_t start = pen_vertex(&pen);
                const std::int64_t control = add_vertex(&pen, reflect_x, reflect_y);
                const SvgSegment segment = {QUADRATIC, {}, {start, control, -1, add_vertex(&pen, end_x, end_y)}, {}};
                add_segment(&pen, &segment, end_x, end_y);
                break;
            }
            default:  // 'A'
                add_arc(&pen, origin_x + values[5], origin_y + values[6], std::abs(values[0]), std::abs(values[1]),
                        values[2], values[3] != 0.0, values[4] != 0.0);
                break;
        }
        // a vertex `format` would refuse makes the command malformed, like a
        // number that overflows: it draws nothing, and parsing ends before it
        if (pen.beyond) {
            out->vertices.resize(vertices_before);
            out->segments.resize(segments_before);
            pen.vertex = pen_vertex_before;
            return token;
        }
        previous = upper;
    }
    return size;
}

// "00" to "99", so digits are written two at a time
static constexpr char DIGIT_PAIRS[201] =
    "0001020304050607080910111213141516171819"
    "2021222324252627282930313233343536373839"
    "4041424344454647484950515253545556575859"
    "6061626364656667686970717273747576777879"
    "8081828384858687888990919293949596979899";

// the decimal digits of `value`, most significant first, ending at `end`:
// returns where they start, at most 39 digits before it
template <typename Unsigned>
static inline char* write_digits_before(char* end, Unsigned value) {
    while (value >= 100) {
        const auto pair = static_cast<std::size_t>(value % 100);
        value /= 100;
        end -= 2;
        std::memcpy(end, &DIGIT_PAIRS[2 * pair], 2);
    }
    if (value >= 10) {
        end -= 2;
        std::memcpy(end, &DIGIT_PAIRS[2 * static_cast<std::size_t>(value)], 2);
    } else {
        *--end = static_cast<char>('0' + static_cast<int>(value));
    }
    return end;
}

// the most characters `write_number` writes: a sign and 24 more, as fixed
// notation is written only when no longer than scientific
static constexpr std::size_t MOST_NUMBER_CHARACTERS = 25;

// A number at `cursor` with at most `digits` significant digits (1 to 17),
// returning where it ends: Dragonbox's shortest digits that read back as
// exactly the same double (the closest of them, ties to even), rounded half
// away from zero when there are more. That rounds the decimal Python's
// `repr` shows, so 2.675 is 2.68 at three digits, where the double itself,
// 2.67499999..., would round to 2.67. At 17 nothing is rounded, and the
// number reads back exactly. Fixed or scientific notation, whichever is
// shorter, fixed on a tie. Not std::to_chars: where libstdc++ is linked
// statically (manylinux_2_28) it brings formatting for every floating type
// and precision, 128 KB. And independent of the C locale, unlike printf.
// `value` is finite: every number `format` writes passed its checks.
static inline char* write_number(char* cursor, double value, int digits) {
    if (value == 0.0) {
        if (std::signbit(value)) {
            *cursor++ = '-';
        }
        *cursor++ = '0';
        return cursor;
    }
    // the shortest digits: an integer below 2^53 (most coordinates) is its
    // own, less trailing zeros, and anything else is Dragonbox's, which has none
    const bool negative = std::signbit(value);
    const double absolute = std::fabs(value);
    std::uint64_t shortest = 0;
    int decimal_exponent = 0;
    const auto strip_zeros = [&] {
        while (shortest % 10 == 0) {
            shortest /= 10;
            decimal_exponent++;
        }
    };
    if (absolute < 9007199254740992.0 && absolute == std::floor(absolute)) {
        shortest = static_cast<std::uint64_t>(absolute);
        strip_zeros();
    } else {
        const auto decimal = jkj::dragonbox::to_decimal(value);
        shortest = decimal.significand;
        decimal_exponent = decimal.exponent;
    }
    char significand[40];
    char* const significand_end = significand + sizeof(significand);
    const char* significant = write_digits_before(significand_end, shortest);
    int count = static_cast<int>(significand_end - significant);
    if (count > digits) {
        // rounded half away from zero, and written again
        std::uint64_t divisor = 1;
        for (int drop = digits; drop < count; drop++) {
            divisor *= 10;
        }
        shortest = (shortest + (divisor / 2)) / divisor;
        decimal_exponent += count - digits;
        strip_zeros();
        significant = write_digits_before(significand_end, shortest);
        count = static_cast<int>(significand_end - significant);
    }
    const int point = count + decimal_exponent;  // how many digits precede the decimal point
    const int exponent = point - 1;              // in scientific notation
    const int magnitude = exponent < 0 ? -exponent : exponent;
    const int scientific_length = count + (count > 1 ? 1 : 0) + 2 + (magnitude >= 100 ? 3 : 2);
    // fixed: digits then zeros ("12300"), a point among them ("12.3"), or
    // "0." then zeros then digits ("0.00123")
    int fixed_length = 2 - point + count;
    if (point >= count) {
        fixed_length = point;
    } else if (point > 0) {
        fixed_length = count + 1;
    }
    if (negative) {
        *cursor++ = '-';
    }
    if (fixed_length > scientific_length) {
        *cursor++ = significant[0];
        if (count > 1) {
            *cursor++ = '.';
            std::memcpy(cursor, significant + 1, static_cast<std::size_t>(count - 1));
            cursor += count - 1;
        }
        *cursor++ = 'e';
        *cursor++ = exponent < 0 ? '-' : '+';
        if (magnitude < 10) {
            *cursor++ = '0';
        }
        char exponent_digits[4];
        const char* first = write_digits_before(exponent_digits + 4, static_cast<unsigned>(magnitude));
        const auto exponent_count = static_cast<std::size_t>(exponent_digits + 4 - first);
        std::memcpy(cursor, first, exponent_count);
        cursor += exponent_count;
    } else if (point > count) {
        // an integer: its digits, then zeros
        std::memcpy(cursor, significant, static_cast<std::size_t>(count));
        cursor += count;
        std::memset(cursor, '0', static_cast<std::size_t>(point - count));
        cursor += point - count;
    } else if (point > 0) {
        std::memcpy(cursor, significant, static_cast<std::size_t>(point));
        cursor += point;
        if (point < count) {
            *cursor++ = '.';
            std::memcpy(cursor, significant + point, static_cast<std::size_t>(count - point));
            cursor += count - point;
        }
    } else {
        *cursor++ = '0';
        *cursor++ = '.';
        std::memset(cursor, '0', static_cast<std::size_t>(-point));
        cursor += -point;
        std::memcpy(cursor, significant, static_cast<std::size_t>(count));
        cursor += count;
    }
    return cursor;
}

// the most characters one segment writes: a moveto and four `A` commands,
// 30 numbers and their letters and spaces
static constexpr std::size_t MOST_SEGMENT_CHARACTERS = 30 * (MOST_NUMBER_CHARACTERS + 2);

// Write a path as absolute path data, its numbers with at most `digits`
// significant digits (1 to 17, as `write_number`), appending to `out`.
// Returns NULL, or why it refused with the index of the segment in
// `refused` (the segment count when it is `digits`).
//
// A segment that doesn't start on the vertex the previous one ended on gets
// a moveto, even to the same coordinates: that is the only way path data
// starts a new vertex. A line ending on its subpath's first vertex, without
// starting there, is `Z`. ARC and CONIC pieces are `A` commands, split as
// `arc_commands` says. Only the slots and `kind_data` a kind uses are read:
// an ARC's center is not, as its points say where its circle is. Anything
// `parse` produced formats; inputs no path data could produce are refused,
// and every index is checked before it is read. `path` is the caller's own:
// nothing else may change it meanwhile.
static inline const char* format(const SvgPath* path, int digits, std::string* out, std::size_t* refused) {
    const std::size_t vertex_count = path->vertices.size() / 2;
    const std::size_t segment_count = path->segments.size();
    *refused = segment_count;
    if (digits < 1 || digits > 17) {
        return "digits must be from 1 to 17";
    }
    std::size_t used = out->size();
    std::int64_t pen = -1;    // the vertex a reader's pen is on
    std::int64_t start = -1;  // the vertex its subpath started on, where `Z` returns
    for (std::size_t index = 0; index < segment_count; index++) {
        *refused = index;
        const SvgSegment* segment = &path->segments[index];
        const std::uint8_t kind = segment->kind;
        if (kind > CONIC) {
            return "kind is not an SvgKind";
        }
        // every point the kind names, each in range, finite and within the
        // bound: start and end, a curve's control or halfway point, a
        // cubic's second control
        const bool names[4] = {true, kind != LINE, kind == CUBIC, true};
        double coordinates[8] = {};
        for (std::size_t slot = 0; slot < 4; slot++) {
            if (!names[slot]) {
                continue;
            }
            const std::int64_t vertex = segment->points[slot];
            if (vertex < 0 || static_cast<std::uint64_t>(vertex) >= vertex_count) {
                return "points names a vertex out of range";
            }
            const std::size_t row = static_cast<std::size_t>(vertex);
            coordinates[2 * slot] = path->vertices[2 * row];
            coordinates[(2 * slot) + 1] = path->vertices[(2 * row) + 1];
            if (!within(coordinates[2 * slot]) || !within(coordinates[(2 * slot) + 1])) {
                return "vertices must be finite and within \xc2\xb1"
                       "1e30";
            }
        }
        SvgArcCommands arc = {};
        if (kind == ARC || kind == CONIC) {
            const char* refusal =
                arc_commands(kind, &coordinates[0], &coordinates[2], &coordinates[6], segment->kind_data[0], &arc);
            if (refusal != nullptr) {
                return refusal;
            }
        }
        const std::int64_t first = segment->points[0];
        const std::int64_t last = segment->points[3];
        const bool moves = first != pen;
        if (moves) {
            start = first;
        }
        const bool closes = kind == LINE && last == start && first != start;
        pen = last;

        if (out->size() < used + MOST_SEGMENT_CHARACTERS) {
            out->resize(std::max(2 * out->size(), used + MOST_SEGMENT_CHARACTERS));
        }
        char* text = out->data() + used;
        // a command letter then its numbers, compactly: "M0 0 L5 0"
        const auto command = [&](char letter, const double* values, int value_count) {
            if (text != out->data()) {
                *text++ = ' ';
            }
            *text++ = letter;
            for (int value = 0; value < value_count; value++) {
                if (value > 0) {
                    *text++ = ' ';
                }
                text = write_number(text, values[value], digits);
            }
        };
        if (moves) {
            command('M', &coordinates[0], 2);
        }
        if (closes) {
            command('Z', coordinates, 0);
        } else if (kind == LINE) {
            command('L', &coordinates[6], 2);
        } else if (kind == QUADRATIC) {
            const double values[4] = {coordinates[2], coordinates[3], coordinates[6], coordinates[7]};
            command('Q', values, 4);
        } else if (kind == CUBIC) {
            command('C', &coordinates[2], 6);
        } else {
            // in degrees, within (-90, 90]: an ellipse turned half a turn is itself
            const double rotation = std::atan2(arc.doubled_rotation[1], arc.doubled_rotation[0]) * 90.0 / PI;
            for (int piece = 0; piece < arc.count; piece++) {
                const double* end = piece + 1 < arc.count ? arc.ends[piece] : &coordinates[6];
                const double values[7] = {arc.radius_x, arc.radius_y, rotation, 0.0, arc.sweep, end[0], end[1]};
                command('A', values, 7);
            }
        }
        used = static_cast<std::size_t>(text - out->data());
    }
    out->resize(used);
    return nullptr;
}

}  // namespace svgpath
