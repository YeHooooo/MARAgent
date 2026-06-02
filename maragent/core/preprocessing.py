from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
from PIL import Image
from scipy.interpolate import interp1d

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


WINDOW_MIN_HU = -175.0
WINDOW_MAX_HU = 275.0
WINDOW_MIN_COEFF = 0.1584
WINDOW_MAX_COEFF = 0.2448


def normalize_tensor(data: np.ndarray, minmax=(0.0, 1.0)) -> np.ndarray:
    data_min, data_max = minmax
    data = np.clip(data, data_min, data_max)
    data = (data - data_min) / (data_max - data_min) * 255.0
    data = data.astype(np.float32)
    return np.expand_dims(np.transpose(np.expand_dims(data, 2), (2, 0, 1)), 0)


def interpolate_projection(proj: np.ndarray, metal_trace: np.ndarray) -> np.ndarray:
    interpolated = proj.copy()
    for i in range(interpolated.shape[0]):
        metal_positions = np.nonzero(metal_trace[i] == 1)[0]
        non_metal_positions = np.nonzero(metal_trace[i] == 0)[0]
        if len(non_metal_positions) == 0:
            interpolated[i][metal_positions] = 0
            continue
        if len(metal_positions) > 0:
            values = interpolated[i][non_metal_positions]
            fn = interp1d(
                non_metal_positions,
                values,
                bounds_error=False,
                fill_value=(values[0], values[-1]),
            )
            interpolated[i][metal_positions] = fn(metal_positions)
    return interpolated


class TensorPreprocessor:
    def __init__(self, device: Any, image_size: int = 416, physical_max: float = 0.5):
        if torch is None:
            raise RuntimeError("PyTorch is required for tensor preprocessing.")
        self.device = device
        self.image_size = int(image_size)
        self.physical_max = float(physical_max)
        self.ray_trafo = None
        self.fbp = None
        self._try_init_geometry()

    @property
    def has_geometry(self) -> bool:
        return self.ray_trafo is not None

    def prepare(self, input_path: str | Path, preview_path: str | Path) -> Dict[str, Any]:
        input_path = Path(input_path)
        suffix = input_path.suffix.lower()
        if suffix == ".npy":
            return self._prepare_npy(input_path, Path(preview_path))
        return self._prepare_image(input_path, Path(preview_path))

    def _try_init_geometry(self) -> None:
        try:
            from tools.InDuDoNet.deeplesion.build_gemotry import build_gemotry, initialization

            param = initialization()
            self.ray_trafo = build_gemotry(param)
        except Exception as exc:
            print(f"[Preprocess] CT geometry is unavailable; geometry-dependent models may be skipped: {exc}")
            self.ray_trafo = None
            self.fbp = None

    def _to_tensor(self, array: np.ndarray, minmax=(0.0, 1.0)):
        return torch.tensor(normalize_tensor(array, minmax), dtype=torch.float32, device=self.device)

    def _prepare_image(self, image_path: Path, preview_path: Path) -> Dict[str, Any]:
        img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError(f"Cannot read image: {image_path}")
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.resize(img_gray, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)

        preview_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(img_gray).save(preview_path)

        xma = (img_gray.astype(np.float32) / 255.0) * self.physical_max
        sorted_pixels = np.sort(xma.flatten())
        threshold = max(sorted_pixels[int(len(sorted_pixels) * 0.98)], self.physical_max * 0.8)
        mask = (xma >= threshold).astype(np.float32)

        xli_u8 = cv2.inpaint(
            (xma / self.physical_max * 255).astype(np.uint8),
            (mask * 255).astype(np.uint8),
            3,
            cv2.INPAINT_NS,
        )
        xli = (xli_u8.astype(np.float32) / 255.0) * self.physical_max
        return self._make_bundle(xma=xma, xli=xli, mask=mask)

    def _prepare_npy(self, npy_path: Path, preview_path: Path) -> Dict[str, Any]:
        img_hu = np.load(npy_path)
        if img_hu.ndim != 2:
            raise ValueError(f"Expected a 2D NPY slice, got shape {img_hu.shape}: {npy_path}")
        if img_hu.shape != (self.image_size, self.image_size):
            img_hu = np.asarray(
                Image.fromarray(img_hu).resize((self.image_size, self.image_size), Image.BILINEAR)
            )

        preview = np.clip(img_hu, WINDOW_MIN_HU, WINDOW_MAX_HU)
        preview = ((preview - WINDOW_MIN_HU) / (WINDOW_MAX_HU - WINDOW_MIN_HU) * 255.0).astype(np.uint8)
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(preview).save(preview_path)

        img_hu_clip = np.clip(img_hu, -1000, None)
        xma = img_hu_clip / 1000.0 * 0.192 + 0.192
        xma = np.ascontiguousarray(xma, dtype=np.float32)
        metal_threshold = 2500 / 1000.0 * 0.192 + 0.192
        mask = (xma > metal_threshold).astype(np.float32)

        if self.ray_trafo is not None:
            pmetal = np.asarray(self.ray_trafo(mask))
            trace = (pmetal > 0).astype(np.float32)
            sma = np.asarray(self.ray_trafo(xma))
            sli = interpolate_projection(sma, trace).astype(np.float32)
            xli = self._fbp_or_inpaint(sli, mask, preview)
            return self._make_bundle(xma=xma, xli=xli, mask=mask, sma=sma, sli=sli, trace=trace)

        xli_u8 = cv2.inpaint(preview, (mask * 255).astype(np.uint8), 3, cv2.INPAINT_NS)
        xli_hu = (xli_u8.astype(np.float32) / 255.0) * (WINDOW_MAX_HU - WINDOW_MIN_HU) + WINDOW_MIN_HU
        xli = np.clip(xli_hu, -1000, None) / 1000.0 * 0.192 + 0.192
        return self._make_bundle(xma=xma, xli=xli, mask=mask)

    def _fbp_or_inpaint(self, sli: np.ndarray, mask: np.ndarray, preview: np.ndarray) -> np.ndarray:
        if self.fbp is not None:
            return np.asarray(self.fbp(sli), dtype=np.float32)
        xli_u8 = cv2.inpaint(preview, (mask * 255).astype(np.uint8), 3, cv2.INPAINT_NS)
        xli_hu = (xli_u8.astype(np.float32) / 255.0) * (WINDOW_MAX_HU - WINDOW_MIN_HU) + WINDOW_MIN_HU
        return (np.clip(xli_hu, -1000, None) / 1000.0 * 0.192 + 0.192).astype(np.float32)

    def _make_bundle(
        self,
        xma: np.ndarray,
        xli: np.ndarray,
        mask: np.ndarray,
        sma: Optional[np.ndarray] = None,
        sli: Optional[np.ndarray] = None,
        trace: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        mask_4d = np.expand_dims(np.transpose(np.expand_dims(mask, 2), (2, 0, 1)), 0)
        bundle: Dict[str, Any] = {
            "Xma": self._to_tensor(xma),
            "XLI": self._to_tensor(xli),
            "Mask": torch.tensor(mask_4d, dtype=torch.float32, device=self.device),
            "non_mask": torch.tensor(1.0 - mask_4d, dtype=torch.float32, device=self.device),
            "Xprior": self._to_tensor(xli),
            "has_geometry": sma is not None and sli is not None and trace is not None,
        }
        if sma is not None and sli is not None and trace is not None:
            trace_processed = 1.0 - trace.astype(np.float32)
            trace_4d = np.expand_dims(np.transpose(np.expand_dims(trace_processed, 2), (2, 0, 1)), 0)
            bundle.update(
                {
                    "Sma": self._to_tensor(sma, (0.0, 4.0)),
                    "SLI": self._to_tensor(sli, (0.0, 4.0)),
                    "Tr": torch.tensor(trace_4d, dtype=torch.float32, device=self.device),
                }
            )
        else:
            bundle.update({"Sma": None, "SLI": None, "Tr": None})
        return bundle
