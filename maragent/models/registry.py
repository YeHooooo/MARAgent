from __future__ import annotations

import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch

from maragent.config import REPO_ROOT, resolve_repo_path


GEOMETRY_MODELS = {"InDuDoNet", "InDuDoNet+"}


class BaseOpt:
    def __init__(self, model_path: str | Path):
        self.model_dir = str(model_path)
        self.use_GPU = True
        self.gpu_id = "0"
        self.batchSize = 1
        self.num_M = 32
        self.num_Q = 32
        self.T = 3
        self.S = 10
        self.etaM = 1
        self.etaX = 5
        self.padding = 4
        self.inP = 5
        self.sizeP = 9
        self.ifini = 1
        self.cdiv = 1
        self.num_rot = 8
        self.num_channel = 32
        self.eta1 = 1
        self.eta2 = 5
        self.alpha = 0.5
        self.N = 6
        self.Np = 32
        self.d = 32
        self.num_res = 3
        self.Mtau = 1.5


class ModelRunner:
    def __init__(
        self,
        device: Any,
        model_config: Dict[str, Any],
        repo_root: Path = REPO_ROOT,
        tools_root: Optional[str | Path] = None,
    ):
        self.device = device
        self.model_config = model_config
        self.repo_root = Path(repo_root)
        self.tools_root = resolve_repo_path(tools_root or "tools", self.repo_root)
        self.weights = model_config.get("weights", {})
        self.models: Dict[str, torch.nn.Module] = {}
        self._ensure_repo_on_path()

    def load_models(self, model_names: Optional[Iterable[str]] = None) -> None:
        names = list(model_names) if model_names is not None else self._all_configured_models()
        for name in names:
            if name in self.models:
                continue
            try:
                model = self._load_one(name)
            except Exception as exc:
                print(f"[ModelRunner] Failed to load {name}: {exc}")
                model = None
            if model is not None:
                self.models[name] = model.eval()
        print(f"[ModelRunner] Loaded models: {sorted(self.models)}")

    def run_inference(self, model_name: str, tensors: Dict[str, Any]):
        if model_name not in self.models:
            self.load_models([model_name])
        if model_name not in self.models:
            print(f"[ModelRunner] Model unavailable: {model_name}")
            return None
        if model_name in GEOMETRY_MODELS and not tensors.get("has_geometry"):
            print(f"[ModelRunner] Skipping {model_name}: CT geometry tensors are unavailable.")
            return None

        net = self.models[model_name]
        with torch.no_grad():
            if model_name in {"DICDNet", "OSCNet", "OSCNet+"}:
                _, list_x, _ = net(tensors["Xma"], tensors["XLI"], tensors["non_mask"])
                return list_x[-1]
            if model_name == "InDuDoNet":
                list_x, _, _ = net(
                    tensors["Xma"],
                    tensors["XLI"],
                    tensors["Mask"],
                    tensors["Sma"],
                    tensors["SLI"],
                    tensors["Tr"],
                )
                return list_x[-1]
            if model_name == "InDuDoNet+":
                list_x, _, _ = net(
                    tensors["Xma"],
                    tensors["XLI"],
                    tensors["Sma"],
                    tensors["SLI"],
                    tensors["Tr"],
                    tensors["Xprior"],
                )
                return list_x[-1]
            if model_name == "ACDNet":
                _, list_x, _, _, _ = net(tensors["Xma"], tensors["XLI"], tensors["non_mask"])
                return list_x[-1]
            if model_name in {"ADN", "SemiMAR", "calimar_gan"}:
                return self._run_unsupervised(model_name, net, tensors)
        return None

    @staticmethod
    def post_process(output_tensor) -> Any:
        output = torch.clamp(output_tensor / 255.0, 0, 0.5) / 0.5
        return output.squeeze().detach().cpu().numpy() * 255.0

    def _run_unsupervised(self, model_name: str, net: torch.nn.Module, tensors: Dict[str, Any]):
        xma_255 = tensors["Xma"] * 2.0
        xli_255 = tensors["XLI"] * 2.0
        input_norm = (xma_255 / 127.5) - 1.0

        if model_name == "ADN":
            result = net.forward1(input_norm)
            out = result[1] if isinstance(result, (list, tuple)) else result
        elif model_name == "SemiMAR":
            out = net.forward2(input_norm) if hasattr(net, "forward2") else net(input_norm)
        else:
            attention = (xli_255 - xma_255).clamp(min=0)
            if attention.max() > 0:
                attention = (attention / attention.max()) * 255.0
            attention_norm = (attention / 127.5) - 1.0
            input_6c = torch.cat(
                [input_norm, input_norm, input_norm, attention_norm, attention_norm, attention_norm],
                dim=1,
            )
            outputs = net(input_6c)
            tensor = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
            out = tensor[:, 1:2, :, :] if tensor.shape[1] >= 2 else tensor[:, 0:1, :, :]
        return ((out + 1.0) / 2.0) * 127.5

    def _load_one(self, name: str):
        path = self._weight_path(name)
        if path is None or not path.exists():
            print(f"[ModelRunner] Missing checkpoint for {name}: {path}")
            return None

        if name == "DICDNet":
            from tools.DICDNet.dicdnet import DICDNet

            opt = BaseOpt(path)
            opt.T = 3
            net = DICDNet(opt).to(self.device)
        elif name == "OSCNet":
            from tools.OSCNet.network.oscnet import OSCNet

            opt = BaseOpt(path)
            opt.num_M = 4
            opt.num_rot = 8
            net = OSCNet(opt).to(self.device)
        elif name == "OSCNet+":
            from tools.OSCNet.network.oscnetplus import OSCNetplus

            opt = BaseOpt(path)
            opt.num_M = 4
            opt.cdiv = 10
            net = OSCNetplus(opt).to(self.device)
        elif name == "InDuDoNet":
            from tools.InDuDoNet.network.indudonet import InDuDoNet

            opt = BaseOpt(path)
            opt.T = 4
            net = InDuDoNet(opt).to(self.device)
        elif name == "InDuDoNet+":
            from tools.InDuDoNet_plus.network.indudonet_plus import InDuDoNet_plus

            opt = BaseOpt(path)
            opt.T = 4
            net = InDuDoNet_plus(opt).to(self.device)
        elif name == "ACDNet":
            from tools.ACDNet.acdnet import ACDNet

            opt = BaseOpt(path)
            opt.N = 6
            opt.Np = 32
            opt.d = 32
            opt.num_res = 3
            opt.T = 10
            opt.Mtau = 1.5
            net = ACDNet(opt).to(self.device)
        elif name == "ADN":
            from tools.adn.adn.adn.networks.adn import ADN

            net = ADN(
                input_ch=1,
                base_ch=64,
                num_down=2,
                num_residual=4,
                num_sides=3,
                res_norm="instance",
                down_norm="instance",
                up_norm="layer",
                fuse=True,
            ).to(self.device)
        elif name == "SemiMAR":
            from tools.SemiMAR.SemiMAR.SemiMAR.networks.SemiMAR import ADN

            net = ADN(
                input_ch=1,
                base_ch=64,
                num_down=2,
                num_residual=4,
                num_sides=3,
                res_norm="instance",
                down_norm="instance",
                up_norm="layer",
                fuse=True,
            ).to(self.device)
        elif name == "calimar_gan":
            calimar_root = self.tools_root / "calimar"
            sys.path.insert(0, str(calimar_root))
            try:
                from models.networks import define_G
            finally:
                if str(calimar_root) in sys.path:
                    sys.path.remove(str(calimar_root))
            net = define_G(input_nc=3, output_nc=3, ngf=64, netG="calimar_gan_A", gpu_ids=[]).to(self.device)
        else:
            raise ValueError(f"Unknown model: {name}")

        state = self._load_checkpoint(path)
        net.load_state_dict(state, strict=False)
        return net

    def _load_checkpoint(self, path: Path) -> OrderedDict:
        checkpoint = torch.load(path, map_location=self.device)
        state = checkpoint
        if isinstance(checkpoint, dict):
            for key in ["model_g", "model_state_dict", "state_dict", "netG", "model"]:
                if key in checkpoint:
                    state = checkpoint[key]
                    break
        return OrderedDict((k.replace("module.", ""), v) for k, v in state.items())

    def _weight_path(self, name: str) -> Optional[Path]:
        raw = self.weights.get(name)
        if not raw:
            return None
        return resolve_repo_path(raw, self.repo_root)

    def _all_configured_models(self):
        return list(self.model_config.get("supervised", [])) + list(self.model_config.get("unsupervised", []))

    def _ensure_repo_on_path(self) -> None:
        for path in [self.repo_root, self.tools_root.parent]:
            text = str(path)
            if text not in sys.path:
                sys.path.insert(0, text)
