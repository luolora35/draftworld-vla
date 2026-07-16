from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyTransformerJEPAPredictor(nn.Module):
    def __init__(
        self,
        visual_dim: int = 2048,
        state_dim: int = 32,
        action_dim: int = 32,
        hidden_dim: int = 512,
        nhead: int = 8,
        layers: int = 2,
    ):
        super().__init__()

        self.context_proj = nn.Linear(visual_dim + state_dim, hidden_dim)
        self.action_proj = nn.Linear(action_dim, hidden_dim)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=layers)

        self.delta_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, visual_dim),
        )

    def forward(self, z_t, state_t, actions):
        ctx = torch.cat([z_t, state_t], dim=-1)
        ctx_tok = self.context_proj(ctx).unsqueeze(1)
        act_tok = self.action_proj(actions)

        x = torch.cat([ctx_tok, act_tok], dim=1)
        x = self.transformer(x)

        delta = self.delta_head(x[:, 1:, :])
        z_pred = z_t.unsqueeze(1) + delta
        return F.normalize(z_pred, dim=-1)


class JEPAShadowLogger:
    def __init__(self):
        self.enabled = os.environ.get("JEPA_SHADOW", "0") == "1"
        self.step = 0
        self.prev_pred = None
        self.prev_meta: dict[str, Any] | None = None

        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if os.environ.get("JEPA_AMP", "1") == "1" else torch.float32

        if not self.enabled:
            return

        ckpt_path = os.environ.get("JEPA_CKPT", "")
        if not ckpt_path:
            raise RuntimeError("JEPA_SHADOW=1 but JEPA_CKPT is empty")

        self.log_dir = Path(os.environ.get("JEPA_LOG_DIR", "/workspace/jepa_logs/shadow_run"))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / "jepa_shadow.jsonl"

        ckpt = torch.load(ckpt_path, map_location="cpu")
        cfg = ckpt.get("args", {})

        self.future_steps = int(os.environ.get("JEPA_FUTURE_STEPS", cfg.get("future_steps", 4)))
        self.action_dim = int(cfg.get("action_dim", 32))

        self.model = TinyTransformerJEPAPredictor(
            visual_dim=int(cfg.get("visual_dim", 2048)),
            state_dim=int(cfg.get("state_dim", 32)),
            action_dim=int(cfg.get("action_dim", 32)),
            hidden_dim=int(cfg.get("hidden_dim", 512)),
            nhead=int(cfg.get("nhead", 8)),
            layers=int(cfg.get("layers", 2)),
        ).to(self.device)

        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

        if self.dtype == torch.float16:
            self.model.half()

        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "event": "init",
                "time": time.time(),
                "ckpt": ckpt_path,
                "future_steps": self.future_steps,
                "device": str(self.device),
                "dtype": str(self.dtype),
            }) + "\n")

        print(
            f"[JEPA SHADOW] enabled ckpt={ckpt_path} "
            f"log={self.log_path} K={self.future_steps}",
            flush=True,
        )

    @staticmethod
    def _prompt_hash(prompt: str) -> str:
        if not prompt:
            return ""
        return hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:10]

    @torch.no_grad()
    def observe(
        self,
        *,
        z_t: torch.Tensor,
        state_t: torch.Tensor,
        actions: torch.Tensor,
        timing: dict[str, float] | None = None,
        prompt: str = "",
        phase: str = "unknown",
    ) -> dict[str, Any] | None:
        if not self.enabled or self.model is None:
            return None

        if z_t.ndim == 1:
            z_t = z_t.unsqueeze(0)
        if state_t.ndim == 1:
            state_t = state_t.unsqueeze(0)
        if actions.ndim == 2:
            actions = actions.unsqueeze(0)

        z_t = z_t.detach().to(self.device, dtype=torch.float32)
        state_t = state_t.detach().to(self.device, dtype=torch.float32)
        actions = actions.detach().to(self.device, dtype=torch.float32)

        actions = actions[:, : self.future_steps, : self.action_dim]

        realized: dict[str, float] = {}

        if self.prev_pred is not None:
            z_now = F.normalize(z_t, dim=-1)

            prev_pred = self.prev_pred.to(self.device, dtype=torch.float32)
            h = int(prev_pred.shape[1])

            z_now_exp = z_now.unsqueeze(1).expand(-1, h, -1)
            scores = 1.0 - F.cosine_similarity(prev_pred, z_now_exp, dim=-1)

            realized["jepa_realized_h1"] = float(scores[:, 0].mean().item())
            realized["jepa_realized_min"] = float(scores.min(dim=1).values.mean().item())
            realized["jepa_realized_mean"] = float(scores.mean().item())
            realized["prev_step"] = float(self.prev_meta.get("step", -1) if self.prev_meta else -1)

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=(self.device.type == "cuda" and self.dtype == torch.float16),
        ):
            z_pred = self.model(
                z_t.to(dtype=self.dtype),
                state_t.to(dtype=self.dtype),
                actions.to(dtype=self.dtype),
            )

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        jepa_ms = (time.perf_counter() - t0) * 1000.0

        row: dict[str, Any] = {
            "event": "step",
            "step": int(self.step),
            "time": time.time(),
            "phase": phase,
            "prompt_hash": self._prompt_hash(prompt),
            "jepa_ms": float(jepa_ms),
            "z_norm": float(z_t.norm(dim=-1).mean().item()),
            "pred_norm": float(z_pred.float().norm(dim=-1).mean().item()),
        }

        row.update(realized)

        if timing is not None:
            for k in [
                "total_ms",
                "encoder_ms",
                "vlm_prefill_ms",
                "decoder_ms",
                "draft_ms",
                "action_verify_ms",
                "accepted_prefix_len",
                "accepted_prefix_len_mean",
                "radius_dist",
                "scheduled_full_fallback",
                "used_full_fallback",
                "is_full_pipeline_round",
            ]:
                if k in timing:
                    try:
                        row[k] = float(timing[k])
                    except Exception:
                        pass

        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

        self.prev_pred = z_pred.detach().float()
        self.prev_meta = {
            "step": int(self.step),
            "phase": phase,
            "time": row["time"],
        }
        self.step += 1

        return row