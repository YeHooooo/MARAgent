from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class PerceptionResult:
    body_part: str = "Unknown"
    implant_type: str = "Unknown"
    severity: str = "Medium"
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CandidateResult:
    name: str
    path: Path


@dataclass
class RouteDecision:
    route: str
    model_pool: List[str]
    models_to_run: List[str]
    is_dental: bool
    reason: str
    memory_result: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ClinicalReport:
    report_text: str
    structural_defects: List[str] = field(default_factory=list)
    has_warning: bool = False
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CaseResult:
    case_id: str
    input_path: Path
    preview_path: Path
    route: RouteDecision
    candidates: List[CandidateResult]
    best_model: str
    best_result_path: Path
    selection_reason: str
    diff_path: Path
    report: ClinicalReport
    summary_path: Optional[Path] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        for key in ["input_path", "preview_path", "best_result_path", "diff_path", "summary_path"]:
            if data.get(key) is not None:
                data[key] = str(data[key])
        for candidate in data["candidates"]:
            candidate["path"] = str(candidate["path"])
        return data
