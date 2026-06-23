"""Force the VPR_Tutorial neural descriptors onto the GPU.

Upstream every learned extractor (AlexNet, NetVLAD, PatchNetVLAD, CosPlace,
EigenPlaces) picks its device with::

    if torch.cuda.is_available():
        ... cuda
    elif mps:
        ... mps
    else:
        print('Using CPU')          # <-- silent fallback
        self.device = torch.device("cpu")

When the container is launched without GPU passthrough, ``torch.cuda.is_available()``
returns False and the models quietly run on CPU -- 10-100x slower with no warning.

This build-time patch replaces that silent CPU branch so the learned descriptors
*require* an accelerator: if neither CUDA nor MPS is visible the run aborts with a
message explaining how to pass the GPU into the container. Set ``VPR_ALLOW_CPU=1``
to opt back into CPU. The SAD baseline in feature_extractor_holistic.py has no
device block and is intentionally left untouched.
"""
import pathlib
import sys

ROOT = pathlib.Path("/VPR_Tutorial/feature_extraction")
FILES = [
    "feature_extractor_holistic.py",      # AlexNetConv3Extractor
    "feature_extractor_cosplace.py",      # CosPlace
    "feature_extractor_eigenplaces.py",   # EigenPlaces
    "feature_extractor_patchnetvlad.py",  # NetVLAD + PatchNetVLAD
]

OLD = (
    "            print('Using CPU')\n"
    '            self.device = torch.device("cpu")\n'
)

NEW = (
    "            import os as _os\n"
    "            if _os.environ.get('VPR_ALLOW_CPU') == '1':\n"
    "                print('Using CPU (VPR_ALLOW_CPU=1)')\n"
    '                self.device = torch.device("cpu")\n'
    "            else:\n"
    "                raise RuntimeError(\n"
    "                    'VPR: no CUDA/MPS device visible -- refusing to silently run on '\n"
    "                    'CPU. Launch the container with GPU access (docker: --gpus all; '\n"
    "                    'podman: --runtime=/usr/bin/nvidia-container-runtime '\n"
    "                    '-e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all), '\n"
    "                    'or set VPR_ALLOW_CPU=1 to override.')\n"
)


def main() -> int:
    failed = []
    for name in FILES:
        path = ROOT / name
        text = path.read_text()
        if NEW in text:
            print(f"already patched: {name}")
            continue
        if OLD not in text:
            failed.append(name)
            continue
        path.write_text(text.replace(OLD, NEW, 1))
        print(f"patched: {name}")
    if failed:
        sys.stderr.write(
            "force_gpu.py: CPU-fallback block not found in: "
            + ", ".join(failed)
            + "\nUpstream VPR_Tutorial changed; update force_gpu.py.\n"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
