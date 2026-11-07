// Replaces the header coal's CMake generates: deprecation markers do nothing,
// as vendored code is compiled quietly anyway.
#ifndef COAL_DEPRECATED_HH
#define COAL_DEPRECATED_HH

#define COAL_DEPRECATED
#define COAL_DEPRECATED_MESSAGE(message)

#endif  // COAL_DEPRECATED_HH
