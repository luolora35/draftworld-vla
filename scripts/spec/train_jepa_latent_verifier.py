from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split


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
        return {
            "z_t": self.z_t[idx],
            "state_t": self.state_t[idx],
            "actions": self.actions[idx],
            "z_future": self.z_future[idx],
        }


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
        """
        z_t:      [B, D]
        state_t:  [B, S]
        actions:  [B, K, A]
        return:   [B, K, D]
        """
        ctx = torch.cat([z_t, state_t], dim=-1)
        ctx_tok = self.context_proj(ctx).unsqueeze(1)
        act_tok = self.action_proj(actions)

        x = torch.cat([ctx_tok, act_tok], dim=1)
        x = self.transformer(x)

        action_tokens = x[:, 1:, :]
        delta = self.delta_head(action_tokens)

        z_pred = z_t.unsqueeze(1) + delta
        z_pred = F.normalize(z_pred, dim=-1)
        return z_pred


def compute_losses(z_pred, z_future, z_t, delta_weight: float = 0.25):
    z_future_norm = F.normalize(z_future.detach(), dim=-1)

    cos_loss = 1.0 - F.cosine_similarity(z_pred, z_future_norm, dim=-1)
    cos_loss = cos_loss.mean()

    pred_delta = z_pred - F.normalize(z_t.unsqueeze(1), dim=-1)
    true_delta = z_future_norm - F.normalize(z_t.unsqueeze(1), dim=-1)
    delta_loss = F.mse_loss(pred_delta, true_delta.detach())

    loss = cos_loss + delta_weight * delta_loss
    return loss, cos_loss.detach(), delta_loss.detach()


@torch.no_grad()
def evaluate(model, loader, device, amp: bool):
    model.eval()

    total_loss = 0.0
    total_cos = 0.0
    total_delta = 0.0
    n = 0

    no_action_cos_total = 0.0
    random_cos_total = 0.0

    for batch in loader:
        z_t = batch["z_t"].to(device)
        state_t = batch["state_t"].to(device)
        actions = batch["actions"].to(device)
        z_future = batch["z_future"].to(device)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            z_pred = model(z_t, state_t, actions)
            loss, cos_loss, delta_loss = compute_losses(z_pred, z_future, z_t)

        z_future_norm = F.normalize(z_future, dim=-1)
        z_now = F.normalize(z_t.unsqueeze(1).expand_as(z_future), dim=-1)
        z_rand = F.normalize(torch.randn_like(z_future), dim=-1)

        no_action_cos = 1.0 - F.cosine_similarity(z_now, z_future_norm, dim=-1)
        random_cos = 1.0 - F.cosine_similarity(z_rand, z_future_norm, dim=-1)

        bs = z_t.shape[0]
        total_loss += float(loss.item()) * bs
        total_cos += float(cos_loss.item()) * bs
        total_delta += float(delta_loss.item()) * bs
        no_action_cos_total += float(no_action_cos.mean().item()) * bs
        random_cos_total += float(random_cos.mean().item()) * bs
        n += bs

    return {
        "loss": total_loss / max(n, 1),
        "cos": total_cos / max(n, 1),
        "delta": total_delta / max(n, 1),
        "no_action_cos": no_action_cos_total / max(n, 1),
        "random_cos": random_cos_total / max(n, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-npz", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)

    parser.add_argument("--visual-dim", type=int, default=2048)
    parser.add_argument("--state-dim", type=int, default=32)
    parser.add_argument("--action-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--layers", type=int, default=2)

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--delta-weight", type=float, default=0.25)
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = JEPALatentDataset(args.train_npz)
    val_size = max(1, int(len(dataset) * args.val_ratio))
    train_size = len(dataset) - val_size

    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
    )

    model = TinyTransformerJEPAPredictor(
        visual_dim=args.visual_dim,
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        hidden_dim=args.hidden_dim,
        nhead=args.nhead,
        layers=args.layers,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print("dataset size:", len(dataset))
    print("train size:", train_size)
    print("val size:", val_size)
    print("params:", f"{n_params / 1e6:.3f}M")
    print("device:", device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()

        total_loss = 0.0
        total_cos = 0.0
        total_delta = 0.0
        n = 0

        for batch in train_loader:
            z_t = batch["z_t"].to(device, non_blocking=True)
            state_t = batch["state_t"].to(device, non_blocking=True)
            actions = batch["actions"].to(device, non_blocking=True)
            z_future = batch["z_future"].to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp):
                z_pred = model(z_t, state_t, actions)
                loss, cos_loss, delta_loss = compute_losses(
                    z_pred,
                    z_future,
                    z_t,
                    delta_weight=args.delta_weight,
                )

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            bs = z_t.shape[0]
            total_loss += float(loss.item()) * bs
            total_cos += float(cos_loss.item()) * bs
            total_delta += float(delta_loss.item()) * bs
            n += bs

        train_loss = total_loss / max(n, 1)
        train_cos = total_cos / max(n, 1)
        train_delta = total_delta / max(n, 1)

        val_metrics = evaluate(model, val_loader, device, amp=args.amp)

        print(
            f"epoch {epoch:03d} | "
            f"train_loss={train_loss:.6f} "
            f"train_cos={train_cos:.6f} "
            f"train_delta={train_delta:.6f} | "
            f"val_loss={val_metrics['loss']:.6f} "
            f"val_cos={val_metrics['cos']:.6f} "
            f"val_delta={val_metrics['delta']:.6f} | "
            f"no_action_cos={val_metrics['no_action_cos']:.6f} "
            f"random_cos={val_metrics['random_cos']:.6f}"
        )

        ckpt = {
            "model": model.state_dict(),
            "args": vars(args),
            "epoch": epoch,
            "val_metrics": val_metrics,
        }

        torch.save(ckpt, out_dir / "last.pt")

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(ckpt, out_dir / "best.pt")
            print("saved best:", out_dir / "best.pt")


if __name__ == "__main__":
    main()