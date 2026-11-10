// Linux cgroup files, read with plain system calls into stack buffers: no
// allocation, no stdio, no locks, so the readers can run inside the
// allocation door (src/trimkit.cpp asks how much room a memory limit leaves)
// and the pool's scheduler (which asks how many CPUs a quota allows). What
// can't be read is no limit. Linux only, and included only there.
#pragma once

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace trimkit::cgroup {

constexpr std::uint64_t NO_LIMIT = UINT64_MAX;
constexpr char ROOT[] = "/sys/fs/cgroup";
// the longest cgroup directory kept, with room for a file name after it
constexpr std::size_t DIRECTORY_CAPACITY = 3072;
constexpr std::size_t PATH_CAPACITY = 4096;

// `path`'s contents into `text`, NUL-terminated and cut to `capacity`: how
// many bytes were read, 0 for a file that can't be
static inline std::size_t read_text(const char* path, char* text, std::size_t capacity) {
    std::size_t size = 0;
    text[0] = '\0';
    const int file = ::open(path, O_RDONLY | O_CLOEXEC);
    if (file < 0) {
        return 0;
    }
    for (;;) {
        const ssize_t got = ::read(file, text + size, capacity - 1 - size);
        if (got <= 0) {
            break;
        }
        size += static_cast<std::size_t>(got);
    }
    ::close(file);
    text[size] = '\0';
    return size;
}

// the number after `name` at the start of a line of `path`, or NO_LIMIT: an
// empty name reads the first line, which cgroups may spell `max`
static inline std::uint64_t read_value(const char* path, const char* name) {
    char text[8192];
    read_text(path, text, sizeof(text));
    const std::size_t length = std::strlen(name);
    for (const char* line = text; line != nullptr && *line != '\0';) {
        if (std::strncmp(line, name, length) == 0) {
            char* end = nullptr;
            const unsigned long long value = std::strtoull(line + length, &end, 10);
            return end == line + length ? NO_LIMIT : value;
        }
        const char* newline = std::strchr(line, '\n');
        line = newline == nullptr ? nullptr : newline + 1;
    }
    return NO_LIMIT;
}

// `directory`/`file` into `path`, or an empty path when it wouldn't fit
static inline void join(char* path, std::size_t capacity, const char* directory, const char* file) {
    const std::size_t directory_length = std::strlen(directory);
    const std::size_t file_length = std::strlen(file);
    path[0] = '\0';
    if (directory_length + 1 + file_length < capacity) {
        std::memcpy(path, directory, directory_length);
        path[directory_length] = '/';
        std::memcpy(path + directory_length + 1, file, file_length + 1);
    }
}

// this process's cgroup v2 directory under ROOT, from the `0::/its/path` line
// of /proc/self/cgroup (the file's last on a mixed system), or ROOT itself
static inline void own_directory(char* directory, std::size_t capacity) {
    std::strncpy(directory, ROOT, capacity);
    directory[capacity - 1] = '\0';
    char text[PATH_CAPACITY];
    read_text("/proc/self/cgroup", text, sizeof(text));
    const char* line = std::strstr(text, "0::/");
    if (line == nullptr || (line != text && line[-1] != '\n')) {
        return;
    }
    const std::size_t length = std::strcspn(line + 3, "\n");
    if (sizeof(ROOT) + length < capacity) {
        std::memcpy(directory + sizeof(ROOT) - 1, line + 3, length);
        directory[sizeof(ROOT) - 1 + length] = '\0';
    }
}

// `directory` cut to its parent: false once it is ROOT, which has no parent
static inline bool ascend(char* directory) {
    if (std::strlen(directory) <= sizeof(ROOT) - 1) {
        return false;
    }
    *std::strrchr(directory, '/') = '\0';
    return true;
}

}  // namespace trimkit::cgroup
