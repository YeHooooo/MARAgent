from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

from maragent.schemas import CandidateResult
from maragent.vlm import BaseVLMClient


SELECTION_PROMPT_TEMPLATE = """
Task: Evaluate MAR results and select the best restoration.
Image 1: input CT image with metal artifacts.
{model_mapping}

Select the model that best balances:
1. artifact suppression,
2. boundary clarity,
3. structural continuity,
4. low over-smoothing risk.

Return ONLY JSON:
{{"best_model": "MODEL_NAME", "reason": "short reason"}}
"""


class RestorationAgent:
    def __init__(self, vlm_client: BaseVLMClient):
        self.vlm = vlm_client

    def select_best(
        self,
        input_path: str | Path,
        candidates: Iterable[CandidateResult],
    ) -> Tuple[str, str]:
        candidates = list(candidates)
        if not candidates:
            raise ValueError("No candidate restorations were produced.")
        if len(candidates) == 1:
            return candidates[0].name, "Selected by Smart Route fast path."

        mapping = "\n".join(
            f"Image {idx + 2}: Model '{candidate.name}'" for idx, candidate in enumerate(candidates)
        )
        prompt = SELECTION_PROMPT_TEMPLATE.format(model_mapping=mapping)
        default = {
            "best_model": candidates[0].name,
            "reason": "VLM selection failed; fell back to the first candidate.",
        }
        image_paths = [input_path] + [candidate.path for candidate in candidates]
        data = self.vlm.call_json(prompt, image_paths, default)
        best_model = str(data.get("best_model", candidates[0].name))
        valid_names = {candidate.name for candidate in candidates}
        if best_model not in valid_names:
            best_model = candidates[0].name
        return best_model, str(data.get("reason", "No reason returned."))
