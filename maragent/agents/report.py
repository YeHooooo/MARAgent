from __future__ import annotations

from pathlib import Path

from maragent.schemas import ClinicalReport
from maragent.vlm import BaseVLMClient


REPORT_PROMPT = """
Act as a radiologist assistant for CT Metal Artifact Reduction (MAR).

Images:
1. Original CT image with metal artifacts.
2. Final restored output.
3. Difference map. Bright regions are high-response modifications. If these
   overlap anatomical boundaries, they may indicate tissue loss or over-smoothing.

Generate a safety-aware clinical report. Return ONLY JSON:
{
  "report_text": "artifact reduction quality, preserved structures, and warnings",
  "structural_defects": ["specific anatomical locations if any"],
  "has_warning": true
}
Set has_warning to false and structural_defects to [] if no clear structure loss is visible.
"""


class ReportAgent:
    def __init__(self, vlm_client: BaseVLMClient):
        self.vlm = vlm_client

    def generate(
        self,
        input_path: str | Path,
        output_path: str | Path,
        diff_path: str | Path,
    ) -> ClinicalReport:
        default = {
            "report_text": "Report generation failed.",
            "structural_defects": [],
            "has_warning": False,
        }
        data = self.vlm.call_json(REPORT_PROMPT, [input_path, output_path, diff_path], default)
        defects = data.get("structural_defects", [])
        if not isinstance(defects, list):
            defects = [str(defects)]
        has_warning = bool(data.get("has_warning", bool(defects)))
        return ClinicalReport(
            report_text=str(data.get("report_text", default["report_text"])),
            structural_defects=[str(item) for item in defects],
            has_warning=has_warning,
            raw=data,
        )
