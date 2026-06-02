from __future__ import annotations

from pathlib import Path

from maragent.schemas import PerceptionResult
from maragent.vlm import BaseVLMClient


PERCEPTION_PROMPT = """
Analyze this medical CT image with metal artifacts.
Return ONLY JSON with:
{
  "body_part": "anatomical region",
  "implant_type": "likely implant or metal source",
  "artifact_severity": "Low | Medium | High | Severe"
}

Severity guide:
- Low: thin streaks, anatomy clearly visible.
- Medium: moderate streaks, partial occlusion or local boundary ambiguity.
- High: strong streaks or dark bands with clear diagnostic interference.
- Severe: large structural loss or very strong artifact dominance.
"""


class PerceptionAgent:
    def __init__(self, vlm_client: BaseVLMClient):
        self.vlm = vlm_client

    def analyze(self, image_path: str | Path) -> PerceptionResult:
        default = {"body_part": "Unknown", "implant_type": "Unknown", "artifact_severity": "Medium"}
        data = self.vlm.call_json(PERCEPTION_PROMPT, [image_path], default)
        severity = data.get("artifact_severity") or data.get("severity") or "Medium"
        return PerceptionResult(
            body_part=str(data.get("body_part", "Unknown")),
            implant_type=str(data.get("implant_type", "Unknown")),
            severity=normalize_severity(str(severity)),
            raw=data,
        )


def normalize_severity(value: str) -> str:
    text = (value or "").strip().lower()
    if text in {"low", "mild"}:
        return "Low"
    if text in {"high", "severe"}:
        return "High" if text == "high" else "Severe"
    return "Medium"
