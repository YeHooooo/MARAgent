from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]


DEFAULT_CONFIG: Dict[str, Any] = {
    "paths": {
        "tools_root": "tools",
        "output_dir": "outputs",
        "memory_root": "memory_bank",
    },
    "runtime": {
        "device": "auto",
        "offline": False,
        "image_size": 416,
        "physical_max": 0.5,
        "memory_warmup_threshold": 50,
        "top_k_memory": 5,
    },
    "vlm": {
        "api_key_env": "MARAGENT_API_KEY",
        "base_url_env": "MARAGENT_BASE_URL",
        "model_env": "MARAGENT_MODEL",
        "default_base_url": "",
        "default_model": "",
        "temperature": 0,
        "max_tokens": 1024,
        "timeout": 300,
        "max_retries": 3,
    },
    "models": {
        "supervised": ["DICDNet", "OSCNet", "OSCNet+", "InDuDoNet", "InDuDoNet+", "ACDNet"],
        "unsupervised": ["ADN", "SemiMAR", "calimar_gan"],
        "fast_supervised": "OSCNet+",
        "fast_unsupervised": "calimar_gan",
        "weights": {
            "DICDNet": "tools/DICDNet/pretrain_model/DICDNet_latest.pt",
            "OSCNet": "tools/OSCNet/pretrained_model/model_osc/net_latest.pt",
            "OSCNet+": "tools/OSCNet/pretrained_model/model_oscplus/net_latest.pt",
            "InDuDoNet": "tools/InDuDoNet/pretrained_model/InDuDoNet_latest.pt",
            "InDuDoNet+": "tools/InDuDoNet_plus/pretrained_model/InDuDoNet+_latest.pt",
            "ACDNet": "tools/ACDNet/models/ACDNet_latest.pt",
            "ADN": "tools/adn/adn/runs/mmdental/net_199.pt",
            "SemiMAR": "tools/SemiMAR/SemiMAR/runs/yofo_data/net_199.pt",
            "calimar_gan": "tools/calimar/checkpoints/my_calimar_training/latest_net_G_A.pth",
        },
    },
}


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, dict):
        return {k: expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env(v) for v in value]
    return value


def load_config(config_path: Optional[str | Path] = None) -> Dict[str, Any]:
    config = deepcopy(DEFAULT_CONFIG)
    if config_path:
        path = Path(config_path)
        try:
            import yaml
        except ImportError:
            if path.name != "default.yaml":
                raise RuntimeError("PyYAML is required to load custom YAML config files.")
            return expand_env(config)
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                user_config = yaml.safe_load(f) or {}
            deep_update(config, user_config)
    return expand_env(config)


def resolve_repo_path(path_like: str | Path, root: Path = REPO_ROOT) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (root / path).resolve()
