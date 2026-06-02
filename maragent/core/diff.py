from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageChops


def make_difference_map(
    input_path: str | Path,
    output_path: str | Path,
    save_path: str | Path,
    enhance: int = 3,
) -> Path:
    input_path = Path(input_path)
    output_path = Path(output_path)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(input_path).convert("L") as img_in:
        with Image.open(output_path).convert("L") as img_out:
            if img_in.size != img_out.size:
                img_in = img_in.resize(img_out.size, Image.BILINEAR)
            diff = ImageChops.difference(img_in, img_out)
            diff = Image.eval(diff, lambda x: min(255, x * enhance))
            diff.save(save_path)
    return save_path
