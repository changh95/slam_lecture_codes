#!/bin/bash
# Fetch the two checkpoints ConceptFusion needs into ~/data/concept_fusion (mounted as /weights):
#   sam_vit_h_4b8939.pth                      SAM ViT-H, 2.4 GB  (Meta)
#   hf/hub/models--laion--CLIP-ViT-H-14-...   OpenCLIP ViT-H-14 laion2b_s32b_b79k, 3.9 GB (Hugging Face)
set -euo pipefail
W=${1:-$HOME/data/concept_fusion}
mkdir -p "$W/hf"
if [ ! -s "$W/sam_vit_h_4b8939.pth" ]; then
  curl -fL -o "$W/sam_vit_h_4b8939.pth.part" \
    https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
  mv "$W/sam_vit_h_4b8939.pth.part" "$W/sam_vit_h_4b8939.pth"
fi
# Let open_clip populate the HF cache itself (CPU only, inside the image).
podman run --rm -v "$W":/weights localhost/slam_zero_to_hero:concept_fusion \
  python -c "import open_clip; open_clip.create_model_and_transforms('ViT-H-14', 'laion2b_s32b_b79k'); print('OpenCLIP ViT-H-14 cached')"
ls -la "$W"; du -sh "$W"
