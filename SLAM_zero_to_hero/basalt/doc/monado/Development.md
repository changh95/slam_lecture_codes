# Installation from source

Please open an issue if these steps don't work for you.

## Build and Install Directories

*This step is optional but recommended, and it is assumed in [build step](#build).*

To not clutter your system directories, let's set two environment variables,
`$bsltdeps` and `$bsltinstall` that point to existing empty build and install
directories respectively. These directories will contain everything produced in
this guide besides installed apt dependencies.

```bash
# Edit the paths as preferred
mkdir -p ~/bsltdeps ~/bsltinstall
export bsltdeps=~/bsltdeps bsltinstall=~/bsltinstall
```

Let's now extend our system paths with those.

```bash
export PATH=$bsltinstall/bin:$PATH                                 # for finding basalt binary tools
export PKG_CONFIG_PATH=$bsltinstall/lib/pkgconfig:$PKG_CONFIG_PATH # for compile time pkg-config
export LD_LIBRARY_PATH=$bsltinstall/lib/:$LD_LIBRARY_PATH          # for runtime ld
export LIBRARY_PATH=$bsltinstall/lib/:$LIBRARY_PATH                # for compile time gcc
```

For persistence in different terminal sessions you can add this
commands to your .bashrc for example if you are using bash.
<details>
<summary><code>echo "..." >> .bashrc</code></summary>

```bash
echo "
export bsltdeps=~/bsltdeps
export bsltinstall=~/bsltinstall
export PATH=\$bsltinstall/bin:\$PATH                                 # for finding basalt binary tools
export PKG_CONFIG_PATH=\$bsltinstall/lib/pkgconfig:\$PKG_CONFIG_PATH # for compile time pkg-config
export LD_LIBRARY_PATH=\$bsltinstall/lib/:\$LD_LIBRARY_PATH          # for runtime ld
export LIBRARY_PATH=\$bsltinstall/lib/:\$LIBRARY_PATH                # for compile time gcc
" >> .bashrc
```

</details>

## Build

```bash
cd $bsltdeps
git clone --recursive https://gitlab.freedesktop.org/mateosss/basalt.git
./basalt/scripts/install_deps.sh
cd basalt && mkdir build && cd build
cmake -B build -G Ninja -DCMAKE_INSTALL_PREFIX=$bsltinstall -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBASALT_INSTANTIATIONS_DOUBLE=off -DBUILD_TESTS=off
cmake --build build --target install
```

## Static analysis

Setup clang-format and clang-tidy.

## Run CI

TODO

## Branch and merge request convention

TODO
