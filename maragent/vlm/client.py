from __future__ import annotations

import base64
import json
import os
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
from PIL import Image


def encode_image(image_path: str | Path, max_size: int = 512) -> Optional[str]:
    try:
        with Image.open(image_path) as img:
            if max(img.size) > max_size:
                img.thumbnail((max_size, max_size))
            if img.mode in ("RGBA", "P", "L"):
                img = img.convert("RGB")
            buffer = BytesIO()
            img.save(buffer, format="JPEG", quality=85)
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception as exc:
        print(f"[VLM] Failed to encode image {image_path}: {exc}")
        return None


def extract_json(text: Optional[str], default: Dict[str, Any]) -> Dict[str, Any]:
    if not text:
        return dict(default)
    start = text.find("{")
    end = text.rfind("}") + 1
    if start < 0 or end <= start:
        return dict(default)
    try:
        parsed = json.loads(text[start:end])
    except json.JSONDecodeError:
        return dict(default)
    if not isinstance(parsed, dict):
        return dict(default)
    return parsed


class BaseVLMClient:
    def call_text(self, prompt: str, image_paths: Iterable[str | Path]) -> Optional[str]:
        raise NotImplementedError

    def call_json(
        self,
        prompt: str,
        image_paths: Iterable[str | Path],
        default: Dict[str, Any],
    ) -> Dict[str, Any]:
        return extract_json(self.call_text(prompt, image_paths), default)


class OpenAICompatibleVLMClient(BaseVLMClient):
    def __init__(self, config: Dict[str, Any]):
        try:
            from langchain_core.messages import HumanMessage
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise RuntimeError(
                "langchain-openai is required for VLM mode. Install requirements or run with --offline."
            ) from exc

        api_key = os.getenv(config.get("api_key_env", "MARAGENT_API_KEY"))
        base_url = os.getenv(config.get("base_url_env", "MARAGENT_BASE_URL")) or config.get("default_base_url")
        model = os.getenv(config.get("model_env", "MARAGENT_MODEL")) or config.get("default_model")
        if not api_key or not model:
            raise RuntimeError(
                "Missing VLM configuration. Set MARAGENT_API_KEY and MARAGENT_MODEL, or run with --offline."
            )

        kwargs: Dict[str, Any] = {
            "model": model,
            "temperature": config.get("temperature", 0),
            "openai_api_key": api_key,
            "max_tokens": config.get("max_tokens", 1024),
            "timeout": config.get("timeout", 300),
            "max_retries": config.get("max_retries", 3),
        }
        if base_url:
            kwargs["base_url"] = base_url

        self._human_message_cls = HumanMessage
        self.llm = ChatOpenAI(**kwargs)

    def call_text(self, prompt: str, image_paths: Iterable[str | Path]) -> Optional[str]:
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for path in image_paths:
            encoded = encode_image(path)
            if encoded:
                content.append(
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}}
                )
        try:
            response = self.llm.invoke([self._human_message_cls(content=content)])
            return response.content
        except Exception as exc:
            print(f"[VLM] API call failed: {exc}")
            return None


class HeuristicVLMClient(BaseVLMClient):
    """Deterministic offline substitute used for smoke tests."""

    def call_text(self, prompt: str, image_paths: Iterable[str | Path]) -> Optional[str]:
        paths = [Path(p) for p in image_paths]
        prompt_lower = prompt.lower()
        if "body_part" in prompt_lower and "implant_type" in prompt_lower:
            return json.dumps(self._analyze(paths[0]))
        if "best_model" in prompt_lower:
            model_names = self._extract_model_names(prompt)
            best = model_names[0] if model_names else "Unknown"
            return json.dumps({"best_model": best, "reason": "Offline heuristic selected the first candidate."})
        if "clinical report" in prompt_lower or "safety" in prompt_lower:
            return json.dumps(self._report(paths))
        return "{}"

    def _analyze(self, image_path: Path) -> Dict[str, Any]:
        with Image.open(image_path).convert("L") as img:
            arr = np.asarray(img, dtype=np.float32)
        bright = float((arr > 230).mean())
        dark = float((arr < 20).mean())
        texture = float(arr.std() / 255.0)
        score = bright * 2.0 + dark + texture
        if score > 0.45:
            severity = "High"
        elif score > 0.22:
            severity = "Medium"
        else:
            severity = "Low"
        name = image_path.name.lower()
        is_dental = any(k in name for k in ["dental", "tooth", "teeth", "mandible", "cbct"])
        return {
            "body_part": "Head" if is_dental else "Unknown",
            "implant_type": "Dental Filling" if is_dental else "Metal Implant",
            "artifact_severity": severity,
        }

    def _report(self, image_paths: List[Path]) -> Dict[str, Any]:
        if len(image_paths) >= 3:
            with Image.open(image_paths[2]).convert("L") as img:
                arr = np.asarray(img, dtype=np.float32)
            changed = float((arr > 50).mean())
        else:
            changed = 0.0
        warning = changed > 0.35
        return {
            "report_text": (
                "Offline safety check: large modified regions were detected."
                if warning
                else "Offline safety check: no obvious large structural modification was detected."
            ),
            "structural_defects": ["High-response difference-map region"] if warning else [],
            "has_warning": warning,
        }

    @staticmethod
    def _extract_model_names(prompt: str) -> List[str]:
        names: List[str] = []
        for line in prompt.splitlines():
            if "Model '" in line:
                start = line.find("Model '") + len("Model '")
                end = line.find("'", start)
                if end > start:
                    names.append(line[start:end])
        return names


def build_vlm_client(config: Dict[str, Any], offline: bool = False) -> BaseVLMClient:
    if offline:
        return HeuristicVLMClient()
    return OpenAICompatibleVLMClient(config)
