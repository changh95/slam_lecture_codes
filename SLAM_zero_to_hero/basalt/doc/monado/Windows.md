# Building on Windows

This guide covers setting up and building Basalt on Windows using PowerShell.
Windows support is preliminary and needs improvements.

## Prerequisites

1. **Install essential tools** using `winget`:
   ```powershell
   winget install -e --id Microsoft.VisualStudioCode # If you need vscode
   winget install -e --id Git.Git
   winget install -e --id Kitware.CMake
   winget install -e --id Ninja-build.Ninja
   winget install -e --id LLVM.LLVM
   winget install -e --id Ccache.Ccache
   ```

2. **Install Visual Studio C++ Environment**:
   - Download the **Visual Studio Installer** "Community" from [here](https://visualstudio.microsoft.com/downloads/).

   - Open it and install the **C++ Desktop Development** option to get the RC compiler.

3. **Set Environment Variables**:
   - Add LLVM binaries to your environment variables to make `clang` commands accessible from any terminal session. In particular llvm-objdump
   ```powershell
   [Environment]::SetEnvironmentVariable("Path", $env:Path + ";C:\Program Files\LLVM\bin", [System.EnvironmentVariableTarget]::Machine)
   ```

4. **Verify Installation**:
   - Check if `clang` and `rc.exe` (Resource Compiler) are accessible:
   ```powershell
   & "C:\Program Files\LLVM\bin\clang.exe" --version
   & "C:\Program Files (x86)\Windows Kits\10\bin\<version>\<x64>\rc.exe"
   ```

## Dependency Installation with vcpkg

Install dependencies via `vcpkg`:

1. **Clone and Bootstrap `vcpkg`**:
   ```powershell
   git clone https://github.com/microsoft/vcpkg.git
   cd vcpkg
   .\bootstrap-vcpkg.bat
   ```

2. **Install Libraries**:
   - Use `vcpkg` to install and compile required libraries (this will take a while):
   ```powershell
   .\vcpkg install tbb opencv fmt boost glew lz4
   ```

3. **Integrate `vcpkg` with CMake**:
   ```powershell
   .\vcpkg integrate install
   ```

4. **Check Installed Packages**:
   ```powershell
   .\vcpkg list
   ```

## Clone and Prepare the Repository

1. **Clone the Project**:
   ```powershell
   git clone --recursive https://gitlab.freedesktop.org/mateosss/basalt.git
   ```

2. **Apply Patches**:
   - Apply necessary patches for Windows compatibility (eventually we will remove these):
   ```powershell
   git apply --directory=thirdparty/basalt-headers/ scripts/windows-patches/basalt-headers.patch
   git apply --directory=thirdparty/opengv/ scripts/windows-patches/opengv.patch
   git apply --directory=thirdparty/ros/ros_comm scripts/windows-patches/ros_comm.patch
   git apply scripts/windows-patches/basalt.patch
   ```

## Configure and Build

1. **Configure the Project**:
   ```powershell
   cmake --preset development `
     -DCMAKE_C_COMPILER="C:/Program Files/LLVM/bin/clang.exe" `
     -DCMAKE_CXX_COMPILER="C:/Program Files/LLVM/bin/clang++.exe" `
     -DCMAKE_RC_COMPILER="C:/Program Files (x86)/Windows Kits/10/bin/10.0.22621.0/x86/rc.exe" `
     -DCMAKE_TOOLCHAIN_FILE="C:/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake"
   ```

2. **Build the Project**:
   ```powershell
   cmake --build build
   ```
   > *Tip:* To clean up, delete the `build` directory using:
   ```powershell
   Remove-Item -Recurse -Force build
   ```

## Downloading and Preparing Datasets

To download and extract datasets directly via command line:

1. **Download with `wget`**:
   - Install `wget` on Windows if needed, or use `Invoke-WebRequest`:
   ```powershell
   wget https://huggingface.co/datasets/collabora/monado-slam-datasets/resolve/main/M_monado_datasets/MO_odyssey_plus/MOO_others/MOO11_short_3_backandforth.zip
   ```

2. **Unzip Dataset**:
   ```powershell
   tar -xf MOO11_short_3_backandforth.zip
   ```

## Running the Application

After the build and dataset setup, you can run the executable with the following command:

```powershell
.\build\basalt_vio.exe --show-gui 1 --deterministic 0 --dataset-path .\MOO11_short_3_backandforth\ --dataset-type euroc --cam-calib .\data\msd\msdmo_calib.json --config-path .\data\msd\msdmo_config.json
```
