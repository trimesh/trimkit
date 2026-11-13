// svgpath: parse SVG path data (the `d` attribute) into indexed vertices and
// segments, and write them back as path data.
//
// Header-only, written as plain C-style C++17: structs, functions and loops,
// depending only on the standard library and fast_float. A path becomes two
// flat arrays in absolute coordinates:
//
// - vertices: x, y pairs, every point once in path order. A segment starts on
//   the vertex the previous one ended on, and a closing `Z` ends on its
//   subpath's first vertex, so connectivity and closure are index equality:
//   a segment that doesn't start on the vertex the previous one ended on
//   starts a new subpath.
// - segments: one per drawing command, naming its points by vertex index
//   (start, two control points or an arc's center and mid-sweep point, end)
//   and carrying an arc's radii, rotation and angles. Relative commands are resolved, H/V become
//   lines and S/T reflections become explicit control points.
//
// Nothing is approximated, so consumers decide what to do with each kind.
// `format` is the inverse: parsing what it writes reproduces the arrays it
// was given bit for bit, for anything `parse` produced.
//
// Every function is `static inline`: each file that includes this header
// gets its own private copy, so two different versions can never collide.
//
// Copyright (c) 2026 Michael Dawson-Haggerty, MIT License.
#pragma once

#include <dragonbox.h>
#include <fast_float/fast_float.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace svgpath {

// the values of `Segment::kind`
enum Kind : std::uint8_t { LINE = 0, QUADRATIC = 1, CUBIC = 2, ARC = 3 };

// One segment, 80 bytes, viewed by numpy as a structured array.
struct Segment {
    std::uint8_t kind;
    // nonzero for the line a `Z` drew: a byte, so any value a numpy bool
    // array holds reads safely
    std::uint8_t closes;
    std::uint8_t padding[6];
    // vertex indices: start, then a curve's first control point or an arc's
    // center, then a cubic's second control point or the point halfway along
    // an arc's sweep, then end; -1 where unused. With start and end, an arc's
    // halfway point is the three-point form of a circular arc
    std::int64_t points[4];
    double radius[2];  // arc: radii, scaled up together if too small (F.6.6)
    double rotation;   // arc: x-axis rotation in degrees, as written
    double angles[2];  // arc: start angle and signed sweep in radians
};
static_assert(sizeof(Segment) == 80, "numpy views Segment as 80 bytes");

// A parsed path: `vertices` holds x, y pairs.
struct Path {
    std::vector<double> vertices;
    std::vector<Segment> segments;
};

static const double PI = 3.14159265358979323846;
// no coordinate, or number that makes one, beyond +-1e30: the bound trimkit
// holds every float64 input to (src/numpy.hpp), so what `parse` produces is
// exactly what `format` and everything downstream accept
static constexpr double COORDINATE_LIMIT = 1e30;
static const double NOT_A_NUMBER = std::numeric_limits<double>::quiet_NaN();

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
struct Reader {
    const char* text;
    std::size_t size;
    std::size_t offset;
};

static inline char peek(const Reader* reader) {
    return reader->offset < reader->size ? reader->text[reader->offset] : '\0';
}

// skip whitespace and commas, and return whether anything is left
static inline bool skip_separators(Reader* reader) {
    while (reader->offset < reader->size && is_separator(reader->text[reader->offset])) {
        reader->offset++;
    }
    return reader->offset < reader->size;
}

static inline bool read_number(Reader* reader, double* value) {
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
    if (!parsed || !(std::abs(*value) <= COORDINATE_LIMIT)) {
        return false;
    }
    reader->offset += static_cast<std::size_t>(result.ptr - start);
    return true;
}

// an arc flag is a single character, so "001" is three flags
static inline bool read_flag(Reader* reader, double* value) {
    if (!skip_separators(reader) || (peek(reader) != '0' && peek(reader) != '1')) {
        return false;
    }
    *value = peek(reader) == '1' ? 1.0 : 0.0;
    reader->offset++;
    return true;
}

// a segment with its arc fields NaN, as every kind but an arc leaves them
static inline Segment make_segment(Kind kind, std::int64_t start, std::int64_t control_0, std::int64_t control_1,
                                   std::int64_t end) {
    Segment segment = {};
    segment.kind = kind;
    segment.points[0] = start;
    segment.points[1] = control_0;
    segment.points[2] = control_1;
    segment.points[3] = end;
    segment.radius[0] = NOT_A_NUMBER;
    segment.radius[1] = NOT_A_NUMBER;
    segment.rotation = NOT_A_NUMBER;
    segment.angles[0] = NOT_A_NUMBER;
    segment.angles[1] = NOT_A_NUMBER;
    return segment;
}

// The center parameterization of an arc from (start_x, start_y) to (end_x,
// end_y), SVG F.6.5, written into `segment` (whose rotation is already set)
// and `center`, and the point halfway along its sweep into `middle`. Returns whether it fits in doubles: radii in a ratio beyond
// the range of a double, scaled up to reach both endpoints, describe no
// finite ellipse, and radii still subnormal once scaled up can't be written
// precisely enough to read back as reaching them (such an arc lies within
// 1e-308 of its chord).
static inline bool arc_center(Segment* segment, double* center, double* middle, double start_x, double start_y,
                              double end_x, double end_y, double radius_x, double radius_y, bool large_arc,
                              bool sweep_positive) {
    const double angle = segment->rotation * PI / 180.0;
    const double cos_angle = std::cos(angle);
    const double sin_angle = std::sin(angle);
    const double half_x = (start_x - end_x) / 2.0;
    const double half_y = (start_y - end_y) / 2.0;
    const double rotated_x = cos_angle * half_x + sin_angle * half_y;
    const double rotated_y = -sin_angle * half_x + cos_angle * half_y;

    // F.6.5 written in the ratios rotated / radius rather than in squares of
    // the radii, which underflow or overflow for very small or large radii:
    // with reach = |ratio|, the center is radius * sqrt(1 - reach^2) times the
    // unit vector (ratio_y, -ratio_x) / reach, and every factor stays in range
    const double ratio_x = rotated_x / radius_x;
    const double ratio_y = rotated_y / radius_y;
    const double reach = std::hypot(ratio_x, ratio_y);
    // a chord below the radii's precision (the ratios underflowed, or the
    // endpoints coincide) places no center: the arc is its chord
    if (reach == 0.0) {
        return false;
    }
    // Radii within a few ulps of exactly reaching both endpoints are treated
    // as reaching them: the center is the chord's midpoint and the radii stay
    // as given. Radii scaled up by F.6.6 and written back out land within
    // rounding of exactly 1 when read again, on either side of it, so without
    // this band they would scale again or move the center off the midpoint,
    // and `format` could not round trip them exactly
    const double band = 16.0 * std::numeric_limits<double>::epsilon();
    double root = 0.0;
    if (reach > 1.0 + band) {
        // radii too small to reach both endpoints are scaled up, F.6.6, and
        // then the center is exactly the midpoint of the chord
        radius_x *= reach;
        radius_y *= reach;
        // subnormal radii can't be written back precisely, and radii past the
        // coordinate bound make an ellipse reaching past it: both are lines
        if (radius_x < std::numeric_limits<double>::min() || radius_y < std::numeric_limits<double>::min() ||
            radius_x > COORDINATE_LIMIT || radius_y > COORDINATE_LIMIT) {
            return false;
        }
    } else if (reach < 1.0 - band) {
        root = std::sqrt(1.0 - (reach * reach));
        if (large_arc == sweep_positive) {
            root = -root;
        }
    }
    const double center_x = root * radius_x * (ratio_y / reach);
    const double center_y = -root * radius_y * (ratio_x / reach);

    // unit vectors from the center to the start (u) and to the end (v)
    const double u_x = (rotated_x - center_x) / radius_x;
    const double u_y = (rotated_y - center_y) / radius_y;
    const double v_x = (-rotated_x - center_x) / radius_x;
    const double v_y = (-rotated_y - center_y) / radius_y;
    // the cross product u x v as u x (v - u), the same exactly, since u x u is
    // zero: written directly it cancels to zero for a chord around 1e-17 of
    // the radius or shorter, and the arc loses its direction. (These divide
    // by the radii as possibly scaled up, so they are not -2 * ratio.)
    const double difference_x = -2.0 * (rotated_x / radius_x);
    const double difference_y = -2.0 * (rotated_y / radius_y);
    double sweep = std::atan2(u_x * difference_y - u_y * difference_x, u_x * v_x + u_y * v_y);
    if (!sweep_positive && sweep > 0.0) {
        sweep -= 2.0 * PI;
    } else if (sweep_positive && sweep < 0.0) {
        sweep += 2.0 * PI;
    }

    segment->radius[0] = radius_x;
    segment->radius[1] = radius_y;
    center[0] = cos_angle * center_x - sin_angle * center_y + (start_x + end_x) / 2.0;
    center[1] = sin_angle * center_x + cos_angle * center_y + (start_y + end_y) / 2.0;
    segment->angles[0] = std::atan2(u_y, u_x);
    segment->angles[1] = sweep;
    // The point halfway along the sweep, without trigonometry: v - u turned a
    // quarter turn clockwise points at the middle of the counterclockwise arc
    // from u to v, however long, and away from it for a clockwise one. The
    // difference is the cancellation-free one above, so this holds for tiny
    // chords too; one too short to have a direction leaves the middle NaN,
    // and the arc a line
    const double turn = sweep > 0.0 ? 1.0 : -1.0;
    const double length = std::hypot(difference_x, difference_y);
    const double halfway_x = radius_x * (turn * difference_y / length);
    const double halfway_y = radius_y * (-turn * difference_x / length);
    middle[0] = center[0] + (cos_angle * halfway_x) - (sin_angle * halfway_y);
    middle[1] = center[1] + (sin_angle * halfway_x) + (cos_angle * halfway_y);
    // an arc is drawn only if its center and halfway point lie within the
    // coordinate bound (false for NaN or infinity too), else it is a line
    return std::isfinite(radius_x) && std::isfinite(radius_y) && std::abs(center[0]) <= COORDINATE_LIMIT &&
           std::abs(center[1]) <= COORDINATE_LIMIT && std::abs(middle[0]) <= COORDINATE_LIMIT &&
           std::abs(middle[1]) <= COORDINATE_LIMIT && std::isfinite(segment->angles[0]) && std::isfinite(sweep);
}

// Where a parse is: the current point and its vertex, and the subpath's
// start, where `Z` returns, and its vertex. A vertex is -1 until a segment
// needs it, so a moveto that draws nothing adds nothing.
struct Pen {
    Path* out;
    double x;
    double y;
    std::int64_t vertex;
    double start_x;
    double start_y;
    std::int64_t start_vertex;
    // why the last segment couldn't be kept, which ends parsing
    const char* refusal;
};

// Why `segment` can't be written as path data, or nullptr: the one definition
// of a valid segment. `format` refuses by it, and `parse` keeps only segments
// it accepts, so whatever `parse` produces `format` writes by construction.
// `subpath_start` is the vertex a closing line must return to, and
// `coordinates` receives the x, y of each point the kind writes.
static inline const char* segment_refusal(const double* vertices, std::size_t vertex_count, const Segment* segment,
                                          std::int64_t subpath_start, double coordinates[8]) {
    if (segment->kind > ARC) {
        return "kind is not an SvgKind";
    }
    if (segment->closes != 0 && segment->kind != LINE) {
        return "only a line can close a subpath";
    }
    // every point the kind names (an arc names its center and halfway point,
    // though only its end is written): each in range, finite and within the
    // bound. Slots a kind doesn't name are ignored
    const bool curve = segment->kind != LINE;
    const bool names[4] = {true, curve, curve && segment->kind != QUADRATIC, true};
    for (std::size_t slot = 0; slot < 4; slot++) {
        if (!names[slot]) {
            continue;
        }
        const std::int64_t vertex = segment->points[slot];
        if (vertex < 0 || static_cast<std::uint64_t>(vertex) >= vertex_count) {
            return "points names a vertex out of range";
        }
        const std::size_t row = static_cast<std::size_t>(vertex);
        coordinates[2 * slot] = vertices[2 * row];
        coordinates[(2 * slot) + 1] = vertices[(2 * row) + 1];
        if (!(std::abs(coordinates[2 * slot]) <= COORDINATE_LIMIT) ||
            !(std::abs(coordinates[(2 * slot) + 1]) <= COORDINATE_LIMIT)) {
            return "vertices must be finite and within \xc2\xb1"
                   "1e30";
        }
    }
    if (segment->closes != 0 && segment->points[3] != subpath_start) {
        return "a closing line must end on its subpath's first vertex";
    }
    if (segment->closes != 0 && segment->points[0] == subpath_start) {
        return "a closing line must close something, not start on its subpath's first vertex";
    }
    // every number `format` writes is one `parse` reads back: within the bound
    if (segment->kind == ARC &&
        (!(std::abs(segment->radius[0]) <= COORDINATE_LIMIT) || !(std::abs(segment->radius[1]) <= COORDINATE_LIMIT) ||
         !(std::abs(segment->rotation) <= COORDINATE_LIMIT) || !std::isfinite(segment->angles[1]))) {
        return "an arc's radius and rotation must be finite and within \xc2\xb1"
               "1e30, and its angles finite";
    }
    return nullptr;
}

static inline std::int64_t add_vertex(Path* out, double x, double y) {
    out->vertices.push_back(x);
    out->vertices.push_back(y);
    return static_cast<std::int64_t>(out->vertices.size() / 2) - 1;
}

// the vertex the pen is on, added the first time a segment needs it
static inline std::int64_t pen_vertex(Pen* pen) {
    if (pen->vertex < 0) {
        pen->vertex = add_vertex(pen->out, pen->x, pen->y);
        if (pen->start_vertex < 0) {
            pen->start_vertex = pen->vertex;  // only ever right after a moveto
        }
    }
    return pen->vertex;
}

// Append a segment and move the pen to its end: the only way `parse` adds
// one, and only if `segment_refusal` accepts it. Finite numbers can still
// overflow once added to the pen or reflected about it, and anything else
// `format` would refuse lands here too, in one place.
static inline void add_segment(Pen* pen, const Segment* segment, double end_x, double end_y) {
    double coordinates[8] = {};
    const Path* out = pen->out;
    pen->refusal =
        segment_refusal(out->vertices.data(), out->vertices.size() / 2, segment, pen->start_vertex, coordinates);
    if (pen->refusal != nullptr) {
        return;
    }
    pen->out->segments.push_back(*segment);
    pen->vertex = segment->points[3];
    pen->x = end_x;
    pen->y = end_y;
}

// a straight line from the pen to a new vertex
static inline void add_line(Pen* pen, double end_x, double end_y) {
    const std::int64_t start = pen_vertex(pen);
    const std::int64_t end = add_vertex(pen->out, end_x, end_y);
    const Segment segment = make_segment(LINE, start, -1, -1, end);
    add_segment(pen, &segment, end_x, end_y);
}

// Parse path data, appending to `out`. Returns how many bytes were parsed:
// all of them on success, else the offset of the first malformed token, with
// `out` holding everything before it, as an SVG renderer would draw.
static inline std::size_t parse(const char* text, std::size_t size, Path* out) {
    Reader reader = {text, size, 0};
    Pen pen = {out, 0.0, 0.0, -1, 0.0, 0.0, -1, nullptr};
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
                const std::int64_t start = pen_vertex(&pen);
                Segment segment = make_segment(LINE, start, -1, -1, pen.start_vertex);
                segment.closes = 1;
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
                const std::int64_t control_0 = add_vertex(out, control_x, control_y);
                const std::int64_t control_1 = add_vertex(out, reflect_x, reflect_y);
                const std::int64_t end = add_vertex(out, end_x, end_y);
                const Segment segment = make_segment(CUBIC, start, control_0, control_1, end);
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
                const std::int64_t control_0 = add_vertex(out, reflect_x, reflect_y);
                const std::int64_t end = add_vertex(out, end_x, end_y);
                const Segment segment = make_segment(QUADRATIC, start, control_0, -1, end);
                add_segment(&pen, &segment, end_x, end_y);
                break;
            }
            default: {  // 'A'
                const double end_x = origin_x + values[5];
                const double end_y = origin_y + values[6];
                if (end_x == pen.x && end_y == pen.y) {
                    break;  // an arc that ends where it starts draws nothing, F.6.2
                }
                // an arc with a zero radius is a straight line, F.6.2, and so
                // is one `arc_center` can't describe in normal doubles
                Segment segment = make_segment(ARC, -1, -1, -1, -1);
                segment.rotation = values[2];
                double center[2] = {0.0, 0.0};
                double middle[2] = {0.0, 0.0};
                const double radius_x = std::abs(values[0]);
                const double radius_y = std::abs(values[1]);
                const bool large_arc = values[3] != 0.0;
                const bool sweep_positive = values[4] != 0.0;
                if (radius_x == 0.0 || radius_y == 0.0 ||
                    !arc_center(&segment, center, middle, pen.x, pen.y, end_x, end_y, radius_x, radius_y, large_arc,
                                sweep_positive)) {
                    add_line(&pen, end_x, end_y);
                    break;
                }
                segment.points[0] = pen_vertex(&pen);
                segment.points[1] = add_vertex(out, center[0], center[1]);
                segment.points[2] = add_vertex(out, middle[0], middle[1]);
                segment.points[3] = add_vertex(out, end_x, end_y);
                add_segment(&pen, &segment, end_x, end_y);
                break;
            }
        }
        // a segment `format` would refuse makes the command malformed, like a
        // number that overflows: it draws nothing, and parsing ends before it
        if (pen.refusal != nullptr) {
            out->vertices.resize(vertices_before);
            out->segments.resize(segments_before);
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

// The exact decimal digits of mantissa * 2^shift, a fixed integer below 1e23
// (up to 77 bits), ending at `end`: in plain 64-bit arithmetic on every
// machine, as base 10^9 limbs, least significant first. A limb stays below
// 10^9 < 2^30, so a limb shifted by up to 30 bits plus its carry fits easily
static inline char* write_shifted_before(char* end, std::uint64_t mantissa, int shift) {
    constexpr std::uint64_t BASE = 1000000000;
    constexpr int LIMBS = 3;                                             // 10^27 > 1e23
    std::uint64_t limbs[LIMBS] = {mantissa % BASE, mantissa / BASE, 0};  // mantissa < 2^53 < 10^18
    while (shift > 0) {
        const int step = shift < 30 ? shift : 30;
        std::uint64_t carry = 0;
        for (std::uint64_t& limb : limbs) {
            const std::uint64_t shifted = (limb << step) + carry;
            limb = shifted % BASE;
            carry = shifted / BASE;
        }
        shift -= step;
    }
    int top = LIMBS - 1;
    while (top > 0 && limbs[top] == 0) {
        top--;
    }
    // every limb below the top one is exactly nine digits, zeros included
    for (int i = 0; i < top; i++) {
        char* start = write_digits_before(end, limbs[i]);
        while (end - start < 9) {
            *--start = '0';
        }
        end = start;
    }
    return write_digits_before(end, limbs[top]);
}

// A number in the shortest form that reads back as exactly the same double,
// written character for character as std::to_chars(first, last, value) does:
// Dragonbox's shortest digits (the closest of them, ties to even, as
// std::to_chars chooses), in fixed or scientific notation, whichever is
// shorter, fixed on a tie, and a fixed integer past its shortest digits
// written exactly, as libstdc++ writes 2^60 as 1152921504606846976. Not
// std::to_chars itself: where libstdc++ is linked statically (manylinux_2_28)
// it brings formatting for every floating type and precision, 128 KB. And
// independent of the C locale, unlike printf. `value` is finite: every number
// `format` writes passed `segment_refusal`.
static inline void write_number(std::string* out, double value) {
    if (value == 0.0) {
        out->append(std::signbit(value) ? "-0" : "0");
        return;
    }
    // the shortest digits: an integer below 2^53 (most coordinates) is its
    // own, less trailing zeros, and anything else is Dragonbox's
    const bool negative = std::signbit(value);
    const double absolute = std::fabs(value);
    std::uint64_t shortest = 0;
    int decimal_exponent = 0;
    if (absolute < 9007199254740992.0 && absolute == std::floor(absolute)) {
        shortest = static_cast<std::uint64_t>(absolute);
        while (shortest % 10 == 0) {
            shortest /= 10;
            decimal_exponent++;
        }
    } else {
        const auto decimal = jkj::dragonbox::to_decimal(value);
        shortest = decimal.significand;
        decimal_exponent = decimal.exponent;
    }
    char significand[40];
    char* const significand_end = significand + sizeof(significand);
    const char* digits = write_digits_before(significand_end, shortest);
    const int count = static_cast<int>(significand_end - digits);
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
    // composed here and appended once: at most a sign and 24 characters, as
    // fixed notation is written only when no longer than scientific
    char text[48];
    char* cursor = text;
    if (negative) {
        *cursor++ = '-';
    }
    if (fixed_length > scientific_length) {
        *cursor++ = digits[0];
        if (count > 1) {
            *cursor++ = '.';
            std::memcpy(cursor, digits + 1, static_cast<std::size_t>(count - 1));
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
        // an integer: its exact value rather than its shortest digits padded
        // with zeros (they differ past 2^53). Fixed notation wins only below
        // 1e23, so it is at most 77 bits, a 53-bit mantissa shifted
        int binary_exponent = 0;
        const double fraction = std::frexp(absolute, &binary_exponent);
        const auto mantissa = static_cast<std::uint64_t>(std::ldexp(fraction, 53));
        const int shift = binary_exponent - 53;
        char exact[40];
        const char* first = shift >= 0 ? write_shifted_before(exact + sizeof(exact), mantissa, shift)
                                       : write_digits_before(exact + sizeof(exact), mantissa >> -shift);
        const auto exact_count = static_cast<std::size_t>(exact + sizeof(exact) - first);
        std::memcpy(cursor, first, exact_count);
        cursor += exact_count;
    } else if (point > 0) {
        std::memcpy(cursor, digits, static_cast<std::size_t>(point));
        cursor += point;
        if (point < count) {
            *cursor++ = '.';
            std::memcpy(cursor, digits + point, static_cast<std::size_t>(count - point));
            cursor += count - point;
        }
    } else {
        *cursor++ = '0';
        *cursor++ = '.';
        std::memset(cursor, '0', static_cast<std::size_t>(-point));
        cursor += -point;
        std::memcpy(cursor, digits, static_cast<std::size_t>(count));
        cursor += count;
    }
    out->append(text, cursor);
}

// a command letter then its numbers, compactly: "M0 0 L5 0"
static inline void write_command(std::string* out, char letter, const double* values, int count) {
    if (!out->empty()) {
        out->push_back(' ');
    }
    out->push_back(letter);
    for (int index = 0; index < count; index++) {
        if (index > 0) {
            out->push_back(' ');
        }
        write_number(out, values[index]);
    }
}

// Write segments as absolute path data, appending to `out`. Returns NULL, or
// why it refused with the index of the segment in `refused`.
//
// A segment that doesn't start on the vertex the previous one ended on gets
// a moveto, even to the same coordinates: that is the only way path data
// starts a new vertex. Closing lines are `Z`, and arcs are written with their
// (possibly F.6.6-scaled) radii and the flags their signed sweep implies.
// Anything `parse` produced formats; inputs no path data could produce are
// refused, and every index is checked before it is read.
static inline const char* format(const double* vertices, std::size_t vertex_count, const Segment* segments,
                                 std::size_t segment_count, std::string* out, std::size_t* refused) {
    std::int64_t pen = -1;    // the vertex a reader's pen is on
    std::int64_t start = -1;  // the vertex its subpath started on, where `Z` returns
    for (std::size_t index = 0; index < segment_count; index++) {
        const Segment* segment = &segments[index];
        *refused = index;
        // a segment not continuing from the pen starts a new subpath there
        const std::int64_t subpath_start = segment->points[0] != pen ? segment->points[0] : start;
        double coordinates[8] = {};
        const char* refusal = segment_refusal(vertices, vertex_count, segment, subpath_start, coordinates);
        if (refusal != nullptr) {
            return refusal;
        }
        if (segment->points[0] != pen) {
            write_command(out, 'M', coordinates, 2);
            start = segment->points[0];
        }
        if (segment->closes != 0) {
            write_command(out, 'Z', coordinates, 0);
        } else if (segment->kind == LINE) {
            write_command(out, 'L', &coordinates[6], 2);
        } else if (segment->kind == QUADRATIC) {
            const double values[4] = {coordinates[2], coordinates[3], coordinates[6], coordinates[7]};
            write_command(out, 'Q', values, 4);
        } else if (segment->kind == CUBIC) {
            write_command(out, 'C', &coordinates[2], 6);
        } else {  // ARC: its center and halfway point are derived again from these when read
            const double large_arc = std::abs(segment->angles[1]) > PI ? 1.0 : 0.0;
            const double sweep_positive = segment->angles[1] > 0.0 ? 1.0 : 0.0;
            const double values[7] = {segment->radius[0], segment->radius[1], segment->rotation, large_arc,
                                      sweep_positive,     coordinates[6],     coordinates[7]};
            write_command(out, 'A', values, 7);
        }
        pen = segment->points[3];
    }
    return nullptr;
}

}  // namespace svgpath
