// The one piece of Boost.Assert the span coal bundles uses, so the module
// needs no Boost: the standard assert, compiled out in release builds.
#ifndef TRIMKIT_BOOST_ASSERT_HPP
#define TRIMKIT_BOOST_ASSERT_HPP

#include <cassert>

#define BOOST_ASSERT(condition) assert(condition)
#define BOOST_ASSERT_MSG(condition, message) assert((condition) && (message))

#endif  // TRIMKIT_BOOST_ASSERT_HPP
