// Replaces the header draco's CMake generates: we compile draco's sources
// directly, so this is the whole feature set, a triangle mesh codec for glTF.
#ifndef DRACO_FEATURES_H_
#define DRACO_FEATURES_H_

#define DRACO_MESH_COMPRESSION_SUPPORTED
#define DRACO_NORMAL_ENCODING_SUPPORTED
#define DRACO_STANDARD_EDGEBREAKER_SUPPORTED
#define DRACO_PREDICTIVE_EDGEBREAKER_SUPPORTED
#define DRACO_BACKWARDS_COMPATIBILITY_SUPPORTED

#endif  // DRACO_FEATURES_H_
