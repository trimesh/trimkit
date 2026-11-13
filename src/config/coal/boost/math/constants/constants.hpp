// The one piece of Boost.Math coal uses: pi, so the module needs no Boost.
#ifndef TRIMKIT_BOOST_MATH_CONSTANTS_HPP
#define TRIMKIT_BOOST_MATH_CONSTANTS_HPP

namespace boost {
namespace math {
namespace constants {
template <typename T>
constexpr T pi() {
    return T(3.141592653589793238462643383279502884L);
}
}  // namespace constants
}  // namespace math
}  // namespace boost

#endif  // TRIMKIT_BOOST_MATH_CONSTANTS_HPP
