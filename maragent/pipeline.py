from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
from PIL import Image

from maragent.agents import PerceptionAgent, ReportAgent, RestorationAgent
from maragent.config import REPO_ROOT, load_config, resolve_repo_path
from maragent.core.diff import make_difference_map
from maragent.core.preprocessing import TensorPreprocessor
from maragent.core.router import SmartRouter
from maragent.memory import CaseMemoryBank
from maragent.models import ModelRunner
from maragent.schemas import CandidateResult, CaseResult
from maragent.vlm import build_vlm_client


class MARAgentPipeline:
    def __init__(self, config: Optional[Dict[str, Any]] = None, config_path: Optional[str | Path] = None):
        self.config = config or load_config(config_path)
        self.repo_root = REPO_ROOT
        self.output_dir = resolve_repo_path(self.config["paths"]["output_dir"], self.repo_root)
        self.memory_root = resolve_repo_path(self.config["paths"]["memory_root"], self.repo_root)
        self.tools_root = resolve_repo_path(self.config["paths"].get("tools_root", "tools"), self.repo_root)
        self._ensure_tools_on_path()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        runtime = self.config.get("runtime", {})
        self.device = self._select_device(runtime.get("device", "auto"))
        offline = bool(runtime.get("offline", False))
        self.vlm = build_vlm_client(self.config.get("vlm", {}), offline=offline)
        self.perception_agent = PerceptionAgent(self.vlm)
        self.restoration_agent = RestorationAgent(self.vlm)
        self.report_agent = ReportAgent(self.vlm)
        self.memory = CaseMemoryBank(
            self.memory_root,
            warmup_threshold=runtime.get("memory_warmup_threshold", 50),
        )
        self.router = SmartRouter(
            self.config.get("models", {}),
            top_k=runtime.get("top_k_memory", 5),
        )
        self.preprocessor = TensorPreprocessor(
            self.device,
            image_size=runtime.get("image_size", 416),
            physical_max=runtime.get("physical_max", 0.5),
        )
        self.runner = ModelRunner(
            self.device,
            self.config.get("models", {}),
            repo_root=self.repo_root,
            tools_root=self.tools_root,
        )

    def run_path(self, input_path: str | Path) -> CaseResult:
        input_path = Path(input_path).resolve()
        case_id = input_path.stem
        paths = self._case_dirs()
        preview_path = paths["inputs"] / f"{case_id}_input.png"
        tensors = self.preprocessor.prepare(input_path, preview_path)

        perception = self.perception_agent.analyze(preview_path)
        decision = self.router.route(perception, preview_path, memory_bank=self.memory)
        print(f"[Pipeline] {case_id}: {perception.body_part}, {perception.implant_type}, {perception.severity}")
        print(f"[Pipeline] Route {decision.route}: {decision.models_to_run}")

        candidates = self._run_candidates(case_id, tensors, decision.models_to_run)
        if not candidates:
            raise RuntimeError(f"No candidate restorations were produced for {case_id}.")

        best_model, reason = self.restoration_agent.select_best(preview_path, candidates)
        best_candidate = next((c for c in candidates if c.name == best_model), candidates[0])
        best_model = best_candidate.name

        diff_path = paths["diff"] / f"{case_id}_Diff_{best_model}.png"
        make_difference_map(preview_path, best_candidate.path, diff_path)
        report = self.report_agent.generate(preview_path, best_candidate.path, diff_path)

        best_dest = paths["best"] / f"{case_id}_Best_{best_model}.png"
        shutil.copy2(best_candidate.path, best_dest)

        self.memory.add_case(
            case_id=case_id,
            input_img_path=preview_path,
            meta_info={
                "body_part": perception.body_part,
                "implant_type": perception.implant_type,
                "artifact_severity": perception.severity,
            },
            best_model=best_model,
            best_result_path=best_dest,
            reason=reason,
            clinical_report=report.raw or {
                "report_text": report.report_text,
                "structural_defects": report.structural_defects,
                "has_warning": report.has_warning,
            },
        )

        result = CaseResult(
            case_id=case_id,
            input_path=input_path,
            preview_path=preview_path,
            route=decision,
            candidates=candidates,
            best_model=best_model,
            best_result_path=best_dest,
            selection_reason=reason,
            diff_path=diff_path,
            report=report,
        )
        summary_path = paths["summaries"] / f"{case_id}.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        result.summary_path = summary_path
        return result

    def run_many(self, input_paths: Iterable[str | Path]) -> List[CaseResult]:
        results: List[CaseResult] = []
        for path in input_paths:
            try:
                results.append(self.run_path(path))
            except Exception as exc:
                print(f"[Pipeline] Failed {path}: {exc}")
        return results

    def _run_candidates(self, case_id: str, tensors: Dict[str, Any], model_names: Iterable[str]) -> List[CandidateResult]:
        candidates: List[CandidateResult] = []
        for model_name in model_names:
            output_tensor = self.runner.run_inference(model_name, tensors)
            if output_tensor is None:
                continue
            output_img = self.runner.post_process(output_tensor)
            model_dir = self.output_dir / model_name
            model_dir.mkdir(parents=True, exist_ok=True)
            save_path = model_dir / f"{case_id}.png"
            Image.fromarray(output_img.astype("uint8")).save(save_path)
            candidates.append(CandidateResult(name=model_name, path=save_path))
            print(f"[Pipeline] Executed {model_name}: {save_path}")
        return candidates

    def _case_dirs(self) -> Dict[str, Path]:
        dirs = {
            "inputs": self.output_dir / "Inputs",
            "best": self.output_dir / "Best_Selections",
            "diff": self.output_dir / "Difference_Maps",
            "summaries": self.output_dir / "Case_Summaries",
        }
        for path in dirs.values():
            path.mkdir(parents=True, exist_ok=True)
        return dirs

    @staticmethod
    def _select_device(value: str):
        if value == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(value)

    def _ensure_tools_on_path(self) -> None:
        parent = str(self.tools_root.parent)
        if parent not in sys.path:
            sys.path.insert(0, parent)
