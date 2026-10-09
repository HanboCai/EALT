from pathlib import Path
import sys
import torch
from torch import nn

class OfficialASTClassifier(nn.Module):
    def __init__(self, num_classes, input_fdim=128, input_tdim=251, precision="auto"):
        super().__init__()
        if precision not in ("auto", "bf16", "fp32"):
            raise ValueError("AST precision must be auto, bf16 or fp32")
        vendor = Path(__file__).resolve().parents[1] / "vendor" / "ast"
        sys.path.insert(0, str(vendor))
        try:
            from ast_models import ASTModel
        except ImportError as exc:
            raise RuntimeError("AST implementation not found. Run setup_pretrained_models.sh first.") from exc
        self.network = ASTModel(
            label_dim=num_classes, fstride=10, tstride=10,
            input_fdim=input_fdim, input_tdim=input_tdim,
            imagenet_pretrain=True, audioset_pretrain=True,
            model_size="base384", verbose=False)
        if not hasattr(type(self.network).forward, "__wrapped__"):
            raise RuntimeError("Expected the known official autocast-decorated AST forward")
        self.precision_policy = precision
        self.last_compute_precision = None

    def forward(self, feature):
        if feature.ndim != 3:
            raise ValueError("Expected [batch, mel, time]")
        use_bf16 = feature.is_cuda and self.precision_policy != "fp32" and torch.cuda.is_bf16_supported()
        if self.precision_policy == "bf16" and not use_bf16:
            raise RuntimeError("BF16 explicitly requested on an unsupported device")
        self.last_compute_precision = "bf16" if use_bf16 else "fp32"
        original_forward = type(self.network).forward.__wrapped__
        with torch.cuda.amp.autocast(enabled=use_bf16, dtype=torch.bfloat16):
            output = original_forward(self.network, feature.float().transpose(1, 2))
        return output.float()
