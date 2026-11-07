// Replaces the header coal's CMake generates: we compile coal's sources
// directly into one static module, so nothing is exported or imported, and
// none of its optional dependencies (octomap, qhull, assimp, logging) exist.
#ifndef COAL_CONFIG_HH
#define COAL_CONFIG_HH

#define COAL_DLLAPI
#define COAL_LOCAL

#endif  // COAL_CONFIG_HH
