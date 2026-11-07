// What a draco decode may still ask for, to refuse a buffer's declared sizes
// before touching any of it. On Linux an allocation succeeds whether or not the
// memory is there, within a container's limit too, and the kernel kills the
// process once the pages are touched: there is no error to catch by then.
// Anything unreadable counts as no limit: this may only ever refuse a decode
// that couldn't have fit, and it never waits on anything.
#ifndef DRACO_AVAILABLE_MEMORY_H_
#define DRACO_AVAILABLE_MEMORY_H_

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#elif defined(_WIN32)
extern "C" __declspec(dllimport) int __stdcall GlobalMemoryStatusEx(void* status);
#endif

namespace draco {

constexpr uint64_t kNoMemoryLimit = std::numeric_limits<uint64_t>::max();

#if defined(__linux__)
// the number after `name` on its line of `path`, or kNoMemoryLimit; an empty
// name reads the file's only value, which cgroups may spell `max`
inline uint64_t ReadMemoryValue(const char* path, const char* name) {
  FILE* file = std::fopen(path, "r");
  if (file == nullptr) {
    return kNoMemoryLimit;
  }
  uint64_t value = kNoMemoryLimit;
  const size_t length = std::strlen(name);
  char line[256];
  while (std::fgets(line, sizeof(line), file) != nullptr) {
    if (std::strncmp(line, name, length) == 0) {
      char* end = nullptr;
      const unsigned long long number = std::strtoull(line + length, &end, 10);
      if (end != line + length) {
        value = number;
      }
      break;
    }
  }
  std::fclose(file);
  return value;
}

// what a cgroup's limit leaves: what it holds, less the file cache the kernel
// gives back before it kills anyone
inline uint64_t CgroupRoom(const char* directory, const char* limit_file,
                           const char* usage_file, const char* cache_name) {
  char path[4096];
  std::snprintf(path, sizeof(path), "%s/%s", directory, limit_file);
  const uint64_t limit = ReadMemoryValue(path, "");
  std::snprintf(path, sizeof(path), "%s/%s", directory, usage_file);
  const uint64_t usage = ReadMemoryValue(path, "");
  if (limit == kNoMemoryLimit || usage == kNoMemoryLimit) {
    return kNoMemoryLimit;
  }
  std::snprintf(path, sizeof(path), "%s/memory.stat", directory);
  const uint64_t cache = ReadMemoryValue(path, cache_name);
  const uint64_t held =
      cache == kNoMemoryLimit ? usage : usage - std::min(usage, cache);
  return limit - std::min(limit, held);
}

// this process's cgroup and each one above it, whichever leaves least
inline uint64_t CgroupsRoom() {
  uint64_t room = CgroupRoom("/sys/fs/cgroup/memory", "memory.limit_in_bytes",
                             "memory.usage_in_bytes", "total_inactive_file ");
  char own[3072] = "";
  FILE* file = std::fopen("/proc/self/cgroup", "r");
  if (file != nullptr) {
    char line[3072];
    while (std::fgets(line, sizeof(line), file) != nullptr) {
      if (std::strncmp(line, "0::", 3) == 0) {
        std::snprintf(own, sizeof(own), "%s", line + 3);
        own[std::strcspn(own, "\n")] = '\0';
      }
    }
    std::fclose(file);
  }
  for (;;) {
    char directory[3200];
    std::snprintf(directory, sizeof(directory), "/sys/fs/cgroup%s", own);
    room = std::min(room, CgroupRoom(directory, "memory.max", "memory.current",
                                     "inactive_file "));
    char* last = std::strrchr(own, '/');
    if (last == nullptr) {
      return room;
    }
    *last = '\0';
  }
}
#endif

inline uint64_t AvailableMemory() {
  uint64_t available = kNoMemoryLimit;
#if defined(__linux__)
  const uint64_t kib = ReadMemoryValue("/proc/meminfo", "MemAvailable:");
  if (kib != kNoMemoryLimit) {
    available = kib * 1024;
  }
  available = std::min(available, CgroupsRoom());
#elif defined(__APPLE__)
  uint64_t total = 0;
  size_t size = sizeof(total);
  if (sysctlbyname("hw.memsize", &total, &size, nullptr, 0) == 0) {
    available = total;
  }
#elif defined(_WIN32)
  // MEMORYSTATUSEX: what can still be committed, within the address space
  struct {
    uint32_t length;
    uint32_t load;
    uint64_t total_physical, available_physical;
    uint64_t total_commit, available_commit;
    uint64_t total_virtual, available_virtual;
    uint64_t available_extended;
  } status = {};
  status.length = sizeof(status);
  if (GlobalMemoryStatusEx(&status) != 0) {
    available = std::min(status.available_commit, status.available_virtual);
  }
#endif
  return std::min<uint64_t>(available, std::numeric_limits<size_t>::max());
}

// raised before any of `bytes` is allocated, so nothing has been touched
inline void RefuseOverAvailableMemory(uint64_t bytes) {
  if (bytes > AvailableMemory()) {
    throw std::bad_alloc();
  }
}

}  // namespace draco

#endif  // DRACO_AVAILABLE_MEMORY_H_
