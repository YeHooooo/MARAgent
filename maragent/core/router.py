from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from maragent.schemas import PerceptionResult, RouteDecision


DENTAL_BODY_KEYWORDS = {
    "dental",
    "tooth",
    "teeth",
    "jaw",
    "mandible",
    "maxilla",
    "oral",
    "maxillofacial",
    "maxillo",
    "head",
}

DENTAL_IMPLANT_KEYWORDS = {
    "dental",
    "tooth",
    "teeth",
    "filling",
    "crown",
    "bridge",
    "orthodont",
}


class SmartRouter:
    def __init__(self, model_config: Dict[str, Any], top_k: int = 5):
        self.supervised = list(model_config.get("supervised", []))
        self.unsupervised = list(model_config.get("unsupervised", []))
        self.fast_supervised = model_config.get("fast_supervised", self.supervised[0] if self.supervised else "")
        self.fast_unsupervised = model_config.get(
            "fast_unsupervised", self.unsupervised[0] if self.unsupervised else ""
        )
        self.top_k = top_k

    def route(
        self,
        perception: PerceptionResult,
        image_path: str | Path,
        memory_bank: Optional[Any] = None,
    ) -> RouteDecision:
        is_dental = self._is_dental(perception.body_part, perception.implant_type)
        model_pool = self.unsupervised if is_dental else self.supervised
        fast_model = self.fast_unsupervised if is_dental else self.fast_supervised
        severity = perception.severity

        if severity == "Low":
            models = [fast_model] if fast_model else model_pool[:1]
            return RouteDecision(
                route="fast_restoration",
                model_pool=model_pool,
                models_to_run=models,
                is_dental=is_dental,
                reason=f"Low severity routed to fast expert {models[0] if models else 'N/A'}.",
            )

        if severity in {"High", "Severe"}:
            return RouteDecision(
                route="all_model_race",
                model_pool=model_pool,
                models_to_run=model_pool,
                is_dental=is_dental,
                reason=f"{severity} severity routed to all available experts.",
            )

        memory_result: Dict[str, Any] = {}
        models = []
        if memory_bank is not None:
            memory_result = memory_bank.retrieve_recommendation(
                image_path,
                body_part=perception.body_part,
                implant_type=perception.implant_type,
                top_k=self.top_k,
            )
            suggested = memory_result.get("suggested_model")
            if memory_result.get("mode") == "SEARCH" and suggested in model_pool:
                models.append(suggested)
                if fast_model in model_pool and fast_model not in models:
                    models.append(fast_model)

        if not models:
            models = model_pool
            reason = "Medium severity routed to full pool because memory is warming up."
        else:
            reason = f"Medium severity routed by memory search: {', '.join(models)}."

        return RouteDecision(
            route="memory_search",
            model_pool=model_pool,
            models_to_run=models,
            is_dental=is_dental,
            reason=reason,
            memory_result=memory_result,
        )

    @staticmethod
    def _is_dental(body_part: str, implant_type: str) -> bool:
        body = (body_part or "").lower()
        implant = (implant_type or "").lower()
        return any(k in body for k in DENTAL_BODY_KEYWORDS) or any(
            k in implant for k in DENTAL_IMPLANT_KEYWORDS
        )
