from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image

from maragent.core.router import SmartRouter


class ImageFeatureExtractor:
    """Small offline visual encoder for memory retrieval."""

    def __init__(self, size: int = 64, hist_bins: int = 64):
        self.size = size
        self.hist_bins = hist_bins

    def get_embedding(self, image_path: str | Path) -> np.ndarray:
        try:
            with Image.open(image_path).convert("L") as img:
                img = img.resize((self.size, self.size), Image.BILINEAR)
                arr = np.asarray(img, dtype=np.float32) / 255.0
        except Exception:
            return np.zeros(self.size * self.size + self.hist_bins, dtype=np.float32)

        hist, _ = np.histogram(arr, bins=self.hist_bins, range=(0.0, 1.0), density=True)
        vec = np.concatenate([arr.flatten(), hist.astype(np.float32)])
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec


class CaseMemoryBank:
    def __init__(self, memory_root: str | Path, warmup_threshold: int = 50):
        self.memory_root = Path(memory_root)
        self.dental_root = self.memory_root / "dental"
        self.general_root = self.memory_root / "general"
        self.dental_root.mkdir(parents=True, exist_ok=True)
        self.general_root.mkdir(parents=True, exist_ok=True)
        self.warmup_threshold = int(warmup_threshold)
        self.extractor = ImageFeatureExtractor()
        self.memory: Dict[str, List[Dict[str, Any]]] = {"dental": [], "general": []}
        self.load_memory()

    def load_memory(self) -> None:
        self.memory = {"dental": [], "general": []}
        for category, root in [("dental", self.dental_root), ("general", self.general_root)]:
            for json_path in sorted(root.glob("*.json")):
                try:
                    with json_path.open("r", encoding="utf-8") as f:
                        item = json.load(f)
                    if "input_vector" in item:
                        item["input_vector"] = np.asarray(item["input_vector"], dtype=np.float32)
                    item["_json_path"] = str(json_path)
                    self.memory[category].append(item)
                except Exception as exc:
                    print(f"[Memory] Skipped bad memory file {json_path}: {exc}")

    def reload_memory(self) -> None:
        self.load_memory()

    def get_memory_size(self) -> int:
        return len(self.memory["dental"]) + len(self.memory["general"])

    def retrieve_recommendation(
        self,
        query_img_path: str | Path,
        body_part: str,
        implant_type: str = "",
        top_k: int = 5,
    ) -> Dict[str, Any]:
        category = self._category(body_part, implant_type)
        cases = self.memory[category]
        if len(cases) < self.warmup_threshold:
            return {
                "mode": "WARMUP",
                "category": category,
                "msg": f"Accumulating {category} memory ({len(cases)}/{self.warmup_threshold}).",
            }

        query = self.extractor.get_embedding(query_img_path)
        scored = []
        for item in cases:
            vec = item.get("input_vector")
            if vec is None:
                continue
            sim = float(np.dot(query, vec) / ((np.linalg.norm(query) * np.linalg.norm(vec)) + 1e-8))
            scored.append((sim, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        retrieved = scored[: max(1, min(top_k, len(scored)))]
        suggested_models = [item.get("best_model", "") for _, item in retrieved if item.get("best_model")]
        if not suggested_models:
            return {"mode": "WARMUP", "category": category, "msg": "No valid model votes in memory."}
        suggested = Counter(suggested_models).most_common(1)[0][0]
        return {
            "mode": "SEARCH",
            "category": category,
            "suggested_model": suggested,
            "candidates": sorted(set(suggested_models)),
            "logs": [f"{sim:.3f}: {item.get('case_id')} -> {item.get('best_model')}" for sim, item in retrieved],
        }

    def add_case(
        self,
        case_id: str,
        input_img_path: str | Path,
        meta_info: Dict[str, Any],
        best_model: str,
        best_result_path: str | Path,
        reason: str,
        clinical_report: Dict[str, Any],
        expert_review: str = "pending",
    ) -> Path:
        category = self._category(meta_info.get("body_part", "Unknown"), meta_info.get("implant_type", ""))
        vector = self.extractor.get_embedding(input_img_path)
        item = {
            "case_id": case_id,
            "input_vector": vector.tolist(),
            "body_part": meta_info.get("body_part", "Unknown"),
            "implant_type": meta_info.get("implant_type", "Unknown"),
            "artifact_severity": meta_info.get("artifact_severity") or meta_info.get("severity", "Unknown"),
            "best_model": best_model,
            "best_result_path": str(best_result_path),
            "selection_reason": reason,
            "clinical_report": clinical_report,
            "expert_review": expert_review,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        root = self.dental_root if category == "dental" else self.general_root
        save_path = root / f"case_{case_id}.json"
        with save_path.open("w", encoding="utf-8") as f:
            json.dump(item, f, indent=2, ensure_ascii=False)
        item["input_vector"] = vector
        item["_json_path"] = str(save_path)
        self.memory[category].append(item)
        return save_path

    @staticmethod
    def _category(body_part: str, implant_type: str) -> str:
        return "dental" if SmartRouter._is_dental(body_part, implant_type) else "general"
