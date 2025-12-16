import torch
from torch.utils.data import Dataset

import cv2
import numpy as np

from pathlib import Path


# ============
# CCPD Dataset
# ============

class CCPDDataset(Dataset):
    def __init__(self, root, split_txt, transform=None):
        """
        CCPD dataset loader.

        Directory structure:
        root/
            splits/
                train.txt
                val.txt
                test.txt
            ccpd_base/
                *.jpg
            ...

        @param root: Dataset root directory.
        @param split_txt: One of {"train.txt", "val.txt", "test.txt"}.
        @param transform: Albumentations transform (image + optional bboxes).
        """
        self.root = Path(root)
        self.transform = transform

        with open(self.root / "splits" / split_txt) as f:
            self.img_rel_paths = [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.img_rel_paths)

    def _parse_filename(self, name):
        """
        Parse CCPD filename annotations.
        """
        stem = Path(name).stem
        fields = stem.split("-")
        assert len(fields) == 7, f"Unexpected filename format: {name}"

        # Area ratio
        area = float(fields[0])

        # Tilt degrees
        tilt_h, tilt_v = map(float, fields[1].split("_"))

        # Bounding box: left-up & right-bottom
        lu, rb = fields[2].split("_")
        x1, y1 = map(int, lu.split("&"))
        x2, y2 = map(int, rb.split("&"))
        bbox = [x1, y1, x2, y2]

        # Four vertices (starting from right-bottom, clockwise)
        vertices = []
        for v in fields[3].split("_"):
            x, y = map(int, v.split("&"))
            vertices.append([x, y])
        vertices = np.array(vertices, dtype=np.float32)

        # License plate number (encoded indices)
        plate_indices = list(map(int, fields[4].split("_")))

        # Brightness and blurriness
        brightness = int(fields[5])
        blurriness = int(fields[6])

        return {
            "area": area,
            "tilt": (tilt_h, tilt_v),
            "bbox": bbox,
            "vertices": vertices,
            "plate_indices": plate_indices,
            "brightness": brightness,
            "blurriness": blurriness,
        }

    def __getitem__(self, idx):
        rel_path = self.img_rel_paths[idx]
        img_path = self.root / rel_path
        stem = Path(rel_path).stem

        # Read image (RGB)
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        h, w = img.shape[:2]

        # Parse annotations from filename
        ann = self._parse_filename(img_path.name)

        # Build binary mask from polygon (four vertices)
        mask = np.zeros((h, w), dtype=np.uint8)
        poly = ann["vertices"]  # shape [4,2], float32
        poly_i = np.round(poly).astype(np.int32)
        cv2.fillPoly(mask, [poly_i], 1)

        # Apply transform consistently (image + mask)
        if self.transform is not None:
            out = self.transform(image=img, mask=mask)
            img_out = out["image"]
            mask_out = out["mask"]

            # Ensure [1, H, W] float mask
            if isinstance(mask_out, torch.Tensor):
                mask_out = mask_out.unsqueeze(0).float()
            else:
                mask_out = mask_out[None, ...].astype(np.float32)

            return img_out, mask_out, stem

        # No transform: return numpy arrays
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        mask = mask[None, ...].astype(np.float32)

        return img, mask, stem
