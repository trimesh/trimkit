# The toolchain behind every `uv run scripts` command but `lock` and `wheel`,
# so they run identically on any machine and in CI. Built by
# `uv run scripts image` (and automatically by those commands) from the
# repository root, see Dockerfile.dockerignore: every input is pinned by digest,
# snapshot date or the hash-locked uv.lock.
FROM debian:trixie@sha256:9c6e381421d0f48d1b8133e573f33c185ee16cbf505eda9c4a8a4791b25ea8a6

# clang (every build's compiler) with lld (its linker), its llvm tools,
# sanitizer and libFuzzer runtimes, analyzers, the Vulkan loader with Mesa's
# CPU device (lavapipe, which runs the GPU ray path in the gate) and glslang
# (which `generate` compiles the shader with), git (every command reads the
# files git tracks, never untracked ones), and what CPython needs to build,
# from the Debian archive as it was on the day the base image was built, read
# from the snapshot date the image records: moving the image's digest moves
# its packages with it. Plain http, as the base image has no CA certificates:
# apt verifies every index against the archive keyring, and snapshots' Release
# files have expired
RUN snapshot="$(sed -n 's|^# http://snapshot.debian.org/archive/debian/\([0-9TZ]*\)$|\1|p' \
        /etc/apt/sources.list.d/debian.sources | head -n 1)" \
    && test -n "$snapshot" \
    && sed -i "s|^URIs: http://deb.debian.org/\(.*\)$|URIs: http://snapshot.debian.org/archive/\1/$snapshot|" \
        /etc/apt/sources.list.d/debian.sources \
    && apt-get -o Acquire::Check-Valid-Until=false update \
    && apt-get install -y --no-install-recommends \
        ca-certificates curl git make pkg-config \
        clang clang-format clang-tidy lld llvm libclang-rt-dev cppcheck \
        python3 zlib1g-dev libffi-dev liblzma-dev libbz2-dev \
        libvulkan1 mesa-vulkan-drivers glslang-tools \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.13.7@sha256:96f40d3904aad8d0bd010539001d36c8e2188b981e1334471a90c4a1d93bb490 \
    /uv /uvx /usr/local/bin/

# uv's own state lives in the image, and builds run as the caller's uid with
# the repository mounted at /src, so caches go under the writable build/docker.
# A config file and Python 3.12 on PATH rather than UV_* variables, so the
# image's uv behaves the same however a command is started in it
ENV HOME=/tmp
RUN export UV_CACHE_DIR=/tmp/uv-cache UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    && uv python install --no-bin 3.12 3.15t \
    && ln -s "$(uv python find --managed-python 3.12)" /usr/local/bin/python3.12 \
    && ln -s "$(uv python find --managed-python 3.15t)" /usr/local/bin/python3.15t \
    && rm -rf /tmp/uv-cache && chmod -R a+rX /opt/uv-python \
    && mkdir -p /etc/uv && printf '%s\n' \
        'cache-dir = "/src/build/docker/uv-cache"' \
        'python-downloads = "never"' > /etc/uv/uv.toml

# last, so editing pyproject.toml only re-runs this cheap layer
# ruff and zizmor at the versions and hashes in uv.lock's `lint` group
COPY pyproject.toml uv.lock /opt/tools-project/
RUN cd /opt/tools-project \
    && UV_CACHE_DIR=/tmp/uv-cache UV_PROJECT_ENVIRONMENT=/opt/tools \
       uv sync --frozen --no-install-project --no-default-groups --only-group lint --python 3.12 \
    && rm -rf /tmp/uv-cache \
    && ln -s /opt/tools/bin/ruff /opt/tools/bin/zizmor /usr/local/bin/

WORKDIR /src
