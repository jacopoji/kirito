## Disclaimer

This project is intended for personal and educational use only. Use of automation or botting tools may violate Nexon's Terms of Service and can result in penalties, including permanent account bans. I do not guarantee continued development or maintenance of this project. It may be discontinued at any time. This project is not affiliated with or endorsed by Nexon. Use at your own risk.

## Building the bot (Windows only)

Ensure the following tools and dependencies are installed:
- Rustup
- [Dioxus CLI](https://dioxuslabs.com/learn/0.7/getting_started/#install-the-dioxus-cli)
- LLVM 21.x
- NodeJS
- OpenCV 4 statically linked binaries via `vcpkg install opencv4[contrib,nonfree]:x64-windows-static`

Then build with the following steps:
1. Inside your `%USERPROFILE%/.cargo/config.toml`, add:
```toml
[env]
OPENCV_DISABLE_PROBES = "environment,pkg_config,cmake,vcpkg_cmake"
VCPKGRS_TRIPLET = "x64-windows-static"
VCPKG_ROOT = # Your vcpkg folder
```
2. If it is your first time building, run `npm install` inside `ui` directory
3. Run `dx build --release --package ui` in the root directory

## GPU acceleration (CUDA)

The bot runs several ONNX models (mob detection, OCR for daily quest world-map navigation, etc.).
These are only accelerated on **Nvidia** GPUs via CUDA - there is no DirectML/ROCm fallback, so AMD
and Intel GPUs always run these models on CPU regardless of setup.

If CUDA isn't properly installed, `ort` (the ONNX Runtime bindings) silently falls back to CPU
with no visible error - the bot still runs, just much slower, since CPU inference on these models
can take 1-2 seconds per call instead of being near-instant on GPU. If navigation, detection, or
any OCR-driven feature feels sluggish, check this first.

To set it up:
1. Check the max CUDA version your driver supports by running `nvidia-smi` in a terminal - look
   for `CUDA Version: X.Y` in the header.
2. Install a matching CUDA Toolkit from the
   [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive) (e.g. CUDA 12.4).
3. Install a matching cuDNN build. **NVIDIA's interactive downloader
   ([developer.nvidia.com/cudnn-downloads](https://developer.nvidia.com/cudnn-downloads)) may only
   offer an `arm64` installer for the latest release on Windows** - if `x86_64` isn't offered there,
   get the zip directly from the redistributable archive instead:
   [developer.download.nvidia.com/compute/cudnn/redist/cudnn/windows-x86_64/](https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/windows-x86_64/)
   Pick the `cudnn-windows-x86_64-<version>_cuda12-archive.zip` matching your CUDA Toolkit's major
   version (`cuda11`/`cuda12`/`cuda13`) - **not** an `arm64` package.
4. Extract the zip and copy the contents of its `bin` folder into your CUDA Toolkit's `bin`
   directory (e.g. `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin\`).
5. Verify it worked: in a debug build, open the Debug tab and click **Check CUDA**. It should
   report that the CUDA execution provider initialized successfully. If it instead reports a
   missing DLL (e.g. `cudnn64_9.dll`), double check the CUDA major version and architecture
   (`x86_64`, not `arm64`) actually match what you installed.
