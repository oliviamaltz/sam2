import numpy as np
from PIL import Image
from pathlib import Path

import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# Model setup
ckpt   = "checkpoints/sam2.1_hiera_large.pt"
cfg    = "configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = SAM2ImagePredictor(build_sam2(cfg, ckpt))
gen = SAM2AutomaticMaskGenerator(predictor.model)

# Folders
stim_dir   = Path("test_stimuli")
output_dir = Path("output_masks")
output_dir.mkdir(exist_ok=True)

for img_path in stim_dir.glob("*.jpg"):
    print(f"Processing {img_path.name}…")
    img = np.array(Image.open(img_path).convert("RGB"))

    # Generate ALL masks for this image
    masks = gen.generate(img)

    if not masks:
        print("  ↳ no masks found")
        continue

    # Save each mask separately
    for i, mask_dict in enumerate(masks):
        seg = mask_dict["segmentation"]         # boolean 2D array
        mask_img = Image.fromarray(seg.astype(np.uint8) * 255)
        out_name = f"{img_path.stem}_mask_{i+1}.png"
        mask_img.save(output_dir / out_name)
        print(f"  ↳ saved {out_name}")

