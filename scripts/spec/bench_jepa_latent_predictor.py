import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyTransformerJEPAPredictor(nn.Module):
    """
    Lightweight action-conditioned JEPA-style latent predictor.

    Input:
        z_t:     [B, D]      current environment latent
        state:   [B, S]      robot state
        actions: [B, K, A]   candidate action chunk

    Output:
        z_pred:  [B, K, D]   predicted future latents for every prefix
    """

    def __init__(
        self,
        visual_dim=1024,
        state_dim=7,
        action_dim=7,
        chunk_size=50,
        hidden_dim=512,
        nhead=8,
        layers=2,
    ):
        super().__init__()
        self.visual_dim = visual_dim
        self.chunk_size = chunk_size

        self.context_proj = nn.Linear(visual_dim + state_dim, hidden_dim)
        self.action_proj = nn.Linear(action_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)

        # Predict latent delta for each future prefix.
        self.delta_head = nn.Linear(hidden_dim, visual_dim)

    def forward(self, z_t, state, actions):
        context = torch.cat([z_t, state], dim=-1)
        context_token = self.context_proj(context).unsqueeze(1)  # [B, 1, H]
        action_tokens = self.action_proj(actions)               # [B, K, H]

        tokens = torch.cat([context_token, action_tokens], dim=1)
        out = self.encoder(tokens)

        action_out = out[:, 1:, :]                              # [B, K, H]
        delta = self.delta_head(action_out)                      # [B, K, D]

        # Residual latent prediction: future latent = current latent + predicted delta.
        z_pred = z_t.unsqueeze(1) + delta
        z_pred = F.normalize(z_pred, dim=-1)
        return z_pred


@torch.inference_mode()
def benchmark(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.dtype == "fp16":
        dtype = torch.float16
    elif args.dtype == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    model = TinyTransformerJEPAPredictor(
        visual_dim=args.visual_dim,
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        chunk_size=args.chunk_size,
        hidden_dim=args.hidden_dim,
        nhead=args.nhead,
        layers=args.layers,
    ).to(device=device, dtype=dtype).eval()

    z_t = torch.randn(args.batch_size, args.visual_dim, device=device, dtype=dtype)
    z_t = F.normalize(z_t, dim=-1)

    state = torch.randn(args.batch_size, args.state_dim, device=device, dtype=dtype)
    actions = torch.randn(args.batch_size, args.chunk_size, args.action_dim, device=device, dtype=dtype)

    for _ in range(args.warmup):
        _ = model(z_t, state, actions)

    if device == "cuda":
        torch.cuda.synchronize()

    times = []

    for _ in range(args.runs):
        if device == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            _ = model(z_t, state, actions)
            end.record()

            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        else:
            t0 = time.perf_counter()
            _ = model(z_t, state, actions)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

    times_t = torch.tensor(times)
    num_params = sum(p.numel() for p in model.parameters())

    print("==== JEPA latent predictor latency benchmark ====")
    print(f"device       : {device}")
    print(f"dtype        : {args.dtype}")
    print(f"params       : {num_params / 1e6:.3f} M")
    print(f"batch_size   : {args.batch_size}")
    print(f"visual_dim   : {args.visual_dim}")
    print(f"chunk_size   : {args.chunk_size}")
    print(f"hidden_dim   : {args.hidden_dim}")
    print(f"layers       : {args.layers}")
    print("-----------------------------------------------")
    print(f"mean latency : {times_t.mean().item():.4f} ms")
    print(f"p50 latency  : {times_t.median().item():.4f} ms")
    print(f"p95 latency  : {times_t.quantile(0.95).item():.4f} ms")
    print(f"min latency  : {times_t.min().item():.4f} ms")
    print(f"max latency  : {times_t.max().item():.4f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--visual-dim", type=int, default=1024)
    parser.add_argument("--state-dim", type=int, default=7)
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--runs", type=int, default=500)
    args = parser.parse_args()

    benchmark(args)