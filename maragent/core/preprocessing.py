from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import cv2
import numpy as np
from PIL import Image
from scipy.interpolate import interp1d

try:
    import h5py
except ImportError:  # pragma: no cover
    h5py = None

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


WINDOW_MIN_HU = -175.0
WINDOW_MAX_HU = 275.0
WINDOW_MIN_COEFF = 0.1584
WINDOW_MAX_COEFF = 0.2448
METAL_THRESHOLD_COEFF = 2500 / 1000.0 * 0.192 + 0.192

DEFAULT_H5_IMAGE_KEYS = ("ma_CT", "Xma", "xma", "image", "ct", "data")
DEFAULT_H5_LI_KEYS = ("LI_CT", "XLI", "xli", "LI", "li_CT")
DEFAULT_H5_MASK_KEYS = ("metal_mask", "mask", "Mask", "M", "metal")
DEFAULT_H5_SMA_KEYS = ("ma_sinogram", "Sma", "sma")
DEFAULT_H5_SLI_KEYS = ("LI_sinogram", "SLI", "sli")
DEFAULT_H5_TRACE_KEYS = ("metal_trace", "Tr", "trace")


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
    def __init__(
        self,
        device: Any,
        image_size: int = 416,
        physical_max: float = 0.5,
        h5_image_keys: Optional[Sequence[str]] = None,
        h5_li_keys: Optional[Sequence[str]] = None,
        h5_mask_keys: Optional[Sequence[str]] = None,
        h5_sma_keys: Optional[Sequence[str]] = None,
        h5_sli_keys: Optional[Sequence[str]] = None,
        h5_trace_keys: Optional[Sequence[str]] = None,
    ):
        if torch is None:
            raise RuntimeError("PyTorch is required for tensor preprocessing.")
        self.device = device
        self.image_size = int(image_size)
        self.physical_max = float(physical_max)
        self.h5_image_keys = tuple(h5_image_keys or DEFAULT_H5_IMAGE_KEYS)
        self.h5_li_keys = tuple(h5_li_keys or DEFAULT_H5_LI_KEYS)
        self.h5_mask_keys = tuple(h5_mask_keys or DEFAULT_H5_MASK_KEYS)
        self.h5_sma_keys = tuple(h5_sma_keys or DEFAULT_H5_SMA_KEYS)
        self.h5_sli_keys = tuple(h5_sli_keys or DEFAULT_H5_SLI_KEYS)
        self.h5_trace_keys = tuple(h5_trace_keys or DEFAULT_H5_TRACE_KEYS)
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
        if suffix in {".h5", ".hdf5"}:
            return self._prepare_h5(input_path, Path(preview_path))
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

    def _prepare_h5(self, h5_path: Path, preview_path: Path) -> Dict[str, Any]:
        if h5py is None:
            raise RuntimeError("h5py is required for H5 input preprocessing.")

        with h5py.File(h5_path, "r") as handle:
            xma_raw = self._read_h5_required(handle, self.h5_image_keys, h5_path)
            xli_raw = self._read_h5_optional(handle, self.h5_li_keys)
            mask_raw = self._read_h5_optional(handle, self.h5_mask_keys)
            sma_raw = self._read_h5_optional(handle, self.h5_sma_keys)
            sli_raw = self._read_h5_optional(handle, self.h5_sli_keys)
            trace_raw = self._read_h5_optional(handle, self.h5_trace_keys)

        xma_raw = self._resize_image_array(self._ensure_2d(xma_raw, h5_path, "ma_CT"))
        xma = self._to_physical_coeff(xma_raw)

        preview = self._coeff_to_preview_u8(xma)
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(preview).save(preview_path)

        if mask_raw is not None:
            mask = self._resize_image_array(
                self._ensure_2d(mask_raw, h5_path, "metal mask"),
                nearest=True,
            )
            mask = (mask > 0).astype(np.float32)
        else:
            mask = self._infer_metal_mask(xma)

        if xli_raw is not None:
            xli_raw = self._resize_image_array(self._ensure_2d(xli_raw, h5_path, "LI_CT"))
            xli = self._to_physical_coeff(xli_raw)
        else:
            xli_u8 = cv2.inpaint(preview, (mask * 255).astype(np.uint8), 3, cv2.INPAINT_NS)
            xli = (xli_u8.astype(np.float32) / 255.0) * self.physical_max

        if sma_raw is not None and sli_raw is not None and trace_raw is not None:
            sma = self._ensure_2d(sma_raw, h5_path, "ma_sinogram").astype(np.float32)
            sli = self._ensure_2d(sli_raw, h5_path, "LI_sinogram").astype(np.float32)
            trace = (self._ensure_2d(trace_raw, h5_path, "metal_trace") > 0).astype(np.float32)
            return self._make_bundle(xma=xma, xli=xli, mask=mask, sma=sma, sli=sli, trace=trace)

        if self.ray_trafo is not None:
            pmetal = np.asarray(self.ray_trafo(mask))
            trace = (pmetal > 0).astype(np.float32)
            sma = np.asarray(self.ray_trafo(xma))
            sli = interpolate_projection(sma, trace).astype(np.float32)
            if xli_raw is None:
                xli = self._fbp_or_inpaint(sli, mask, preview)
            return self._make_bundle(xma=xma, xli=xli, mask=mask, sma=sma, sli=sli, trace=trace)

        return self._make_bundle(xma=xma, xli=xli, mask=mask)

    def _read_h5_required(self, handle: Any, keys: Sequence[str], h5_path: Path) -> np.ndarray:
        data = self._read_h5_optional(handle, keys)
        if data is None:
            available = ", ".join(self._list_h5_datasets(handle)) or "<none>"
            expected = ", ".join(keys)
            raise KeyError(f"{h5_path} does not contain any image key from [{expected}]. Available: {available}")
        return data

    def _read_h5_optional(self, handle: Any, keys: Sequence[str]) -> Optional[np.ndarray]:
        for key in keys:
            if key in handle and isinstance(handle[key], h5py.Dataset):
                return handle[key][()]

        key_set = set(keys)
        found: list[np.ndarray] = []

        def visitor(name: str, obj: Any) -> None:
            if found:
                return
            if isinstance(obj, h5py.Dataset) and name.split("/")[-1] in key_set:
                found.append(obj[()])

        handle.visititems(visitor)
        return found[0] if found else None

    def _list_h5_datasets(self, handle: Any) -> list[str]:
        names: list[str] = []

        def visitor(name: str, obj: Any) -> None:
            if isinstance(obj, h5py.Dataset):
                names.append(name)

        handle.visititems(visitor)
        return names

    def _ensure_2d(self, data: np.ndarray, input_path: Path, label: str) -> np.ndarray:
        array = np.asarray(data, dtype=np.float32)
        array = np.squeeze(array)
        if array.ndim == 2:
            return array
        if array.ndim == 3:
            if array.shape[0] in {1, 3, 4}:
                return array[0]
            if array.shape[-1] in {1, 3, 4}:
                return array[..., 0]
        raise ValueError(f"Expected a 2D {label} dataset, got shape {array.shape}: {input_path}")

    def _resize_image_array(self, array: np.ndarray, nearest: bool = False) -> np.ndarray:
        if array.shape == (self.image_size, self.image_size):
            return np.ascontiguousarray(array, dtype=np.float32)
        interpolation = Image.NEAREST if nearest else Image.BILINEAR
        resized = Image.fromarray(array.astype(np.float32)).resize(
            (self.image_size, self.image_size),
            interpolation,
        )
        return np.ascontiguousarray(np.asarray(resized, dtype=np.float32))

    def _to_physical_coeff(self, data: np.ndarray) -> np.ndarray:
        if self._looks_like_hu(data):
            coeff = np.clip(data, -1000, None) / 1000.0 * 0.192 + 0.192
        else:
            coeff = np.clip(data, 0.0, 1.0)
        return np.ascontiguousarray(coeff, dtype=np.float32)

    @staticmethod
    def _looks_like_hu(data: np.ndarray) -> bool:
        finite = data[np.isfinite(data)]
        if finite.size == 0:
            return False
        return float(np.nanmin(finite)) < -10.0 or float(np.nanmax(finite)) > 10.0

    def _coeff_to_preview_u8(self, coeff: np.ndarray) -> np.ndarray:
        clipped = np.clip(coeff, 0.0, self.physical_max)
        return (clipped / self.physical_max * 255.0).astype(np.uint8)

    def _infer_metal_mask(self, xma: np.ndarray) -> np.ndarray:
        if float(np.nanmax(xma)) > METAL_THRESHOLD_COEFF:
            return (xma > METAL_THRESHOLD_COEFF).astype(np.float32)
        sorted_pixels = np.sort(xma.flatten())
        threshold = max(sorted_pixels[int(len(sorted_pixels) * 0.98)], self.physical_max * 0.8)
        return (xma >= threshold).astype(np.float32)

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
