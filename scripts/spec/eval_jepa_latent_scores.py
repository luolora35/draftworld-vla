from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class JEPALatentDataset(Dataset):
    def __init__(self, npz_path: str):
        d = np.load(npz_path)
        self.z_t = torch.from_numpy(d["z_t"].astype(np.float32))
        self.state_t = torch.from_numpy(d["state_t"].astype(np.float32))
        self.actions = torch.from_numpy(d["actions"].astype(np.float32))
        self.z_future = torch.from_numpy(d["z_future"].astype(np.float32))

    def __len__(self):
        return self.z_t.shape[0]

    def __getitem__(self, idx):
        return self.z_t[idx], self.state_t[idx], self.actions[idx], self.z_future[idx]


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


def summarize(name: str, x: np.ndarray):
    print(f"\n{name}")
    print("mean:", float(np.mean(x)))
    print("std :", float(np.std(x)))
    for q in [0, 25, 50, 75, 90, 95, 99, 100]:
        print(f"p{q:02d} :", float(np.percentile(x, q)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--out", type=str, default="")
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
    )

    model.load_state_dict(ckpt["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    ds = JEPALatentDataset(args.npz)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    jepa_scores = []
    no_action_scores = []
    random_scores = []

    with torch.no_grad():
        for z_t, state_t, actions, z_future in loader:
            z_t = z_t.to(device)
            state_t = state_t.to(device)
            actions = actions.to(device)
            z_future = z_future.to(device)

            z_pred = model(z_t, state_t, actions)
            z_future_n = F.normalize(z_future, dim=-1)

            score = 1.0 - F.cosine_similarity(z_pred, z_future_n, dim=-1)
            jepa_scores.append(score.cpu().numpy())

            z_now = F.normalize(z_t.unsqueeze(1).expand_as(z_future), dim=-1)
            no_action = 1.0 - F.cosine_similarity(z_now, z_future_n, dim=-1)
            no_action_scores.append(no_action.cpu().numpy())

            z_rand = F.normalize(torch.randn_like(z_future), dim=-1)
            random = 1.0 - F.cosine_similarity(z_rand, z_future_n, dim=-1)
            random_scores.append(random.cpu().numpy())

    jepa_scores = np.concatenate(jepa_scores, axis=0)
    no_action_scores = np.concatenate(no_action_scores, axis=0)
    random_scores = np.concatenate(random_scores, axis=0)

    summarize("JEPA score: 1 - cos(z_pred, z_future)", jepa_scores.reshape(-1))
    summarize("No-action score: 1 - cos(z_t, z_future)", no_action_scores.reshape(-1))
    summarize("Random score", random_scores.reshape(-1))

    per_window_jepa = jepa_scores.mean(axis=1)
    per_window_no_action = no_action_scores.mean(axis=1)

    summarize("JEPA per-window mean score", per_window_jepa)
    summarize("No-action per-window mean score", per_window_no_action)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out,
            jepa_scores=jepa_scores,
            no_action_scores=no_action_scores,
            random_scores=random_scores,
            per_window_jepa=per_window_jepa,
            per_window_no_action=per_window_no_action,
        )
        print("\nsaved:", out)


if __name__ == "__main__":
    main()