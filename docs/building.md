## Prerequisites

- A C++17 compiler
- CMake (≥ 3.24)
- libxml2.2
    - You can get this on ArchLinux via `pacman -S libxml2-legacy`

### Building

The included `src/build.sh` wipes and rebuilds `src/build/`:

```sh
cd src
./build.sh
```

This runs `cmake -B ./build && cmake --build ./build`. The resulting binary is
`src/build/digits`, and the shared libraries are
`src/build/lib/libdigits/liblibdigits.so` and
`src/build/lib/libdigitspy/liblibdigitspy.so`.
