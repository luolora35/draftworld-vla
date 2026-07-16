from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyTransformerJEPAPredictor(nn.Module):
    def __init__(
        self,
        visual_dim=2048,
        state_dim=32,
        action_dim=32,
        hidden_dim=512,
        nhead=8,
        layers=2,
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
        return F.normalize(z_t.unsqueeze(1) + delta, dim=-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--future-steps", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--runs", type=int, default=1000)
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt.get("args", {})

    model = TinyTransformerJEPAPredictor(
        visual_dim=int(cfg.get("visual_dim", 2048)),
        state_dim=int(cfg.get("state_dim", 32)),
        action_dim=int(cfg.get("action_dim", 32)),
        hidden_dim=int(cfg.get("hidden_dim", 512)),
        nhead=int(cfg.get("nhead", 8)),
        layers=int(cfg.get("layers", 2)),
    ).cuda().eval()

    model.load_state_dict(ckpt["model"])

    B = args.batch_size
    K = args.future_steps

    z_t = torch.randn(B, 2048, device="cuda", dtype=torch.float16 if args.amp else torch.float32)
    state_t = torch.randn(B, 32, device="cuda", dtype=torch.float16 if args.amp else torch.float32)
    actions = torch.randn(B, K, 32, device="cuda", dtype=torch.float16 if args.amp else torch.float32)

    if args.amp:
        model = model.half()

    with torch.no_grad():
        for _ in range(args.warmup):
            _ = model(z_t, state_t, actions)

        torch.cuda.synchronize()

        times = []
        for _ in range(args.runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(z_t, state_t, actions)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

    x = torch.tensor(times)
    print("runs:", args.runs)
    print("batch size:", B)
    print("future steps:", K)
    print("mean latency ms:", float(x.mean()))
    print("p50 latency ms:", float(x.quantile(0.50)))
    print("p95 latency ms:", float(x.quantile(0.95)))
    print("p99 latency ms:", float(x.quantile(0.99)))
    print("max latency ms:", float(x.max()))


if __name__ == "__main__":
    main()