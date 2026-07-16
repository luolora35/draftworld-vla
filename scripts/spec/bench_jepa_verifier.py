import argparse
import time
import torch
import torch.nn as nn


class MLPVerifier(nn.Module):
    def __init__(self, visual_dim=1024, state_dim=7, action_dim=7, chunk_size=12, hidden_dim=512):
        super().__init__()
        self.chunk_size = chunk_size
        input_dim = visual_dim + state_dim + chunk_size * action_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, chunk_size),
        )

    def forward(self, z, state, actions):
        b = z.shape[0]
        x = torch.cat([z, state, actions.reshape(b, -1)], dim=-1)
        return self.net(x)


class GRUVerifier(nn.Module):
    def __init__(self, visual_dim=1024, state_dim=7, action_dim=7, chunk_size=12, hidden_dim=512):
        super().__init__()
        self.chunk_size = chunk_size
        self.context_proj = nn.Linear(visual_dim + state_dim, hidden_dim)
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.score_head = nn.Linear(hidden_dim, 1)

    def forward(self, z, state, actions):
        context = torch.cat([z, state], dim=-1)
        h0 = self.context_proj(context).unsqueeze(0)
        x = self.action_proj(actions)
        out, _ = self.gru(x, h0)
        scores = self.score_head(out).squeeze(-1)
        return scores


class TinyTransformerVerifier(nn.Module):
    def __init__(self, visual_dim=1024, state_dim=7, action_dim=7, chunk_size=12, hidden_dim=512, nhead=8, layers=2):
        super().__init__()
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
        self.score_head = nn.Linear(hidden_dim, 1)

    def forward(self, z, state, actions):
        context = torch.cat([z, state], dim=-1)
        context_token = self.context_proj(context).unsqueeze(1)
        action_tokens = self.action_proj(actions)

        tokens = torch.cat([context_token, action_tokens], dim=1)
        out = self.encoder(tokens)

        action_out = out[:, 1:, :]
        scores = self.score_head(action_out).squeeze(-1)
        return scores


def build_model(kind, visual_dim, state_dim, action_dim, chunk_size, hidden_dim):
    if kind == "mlp":
        return MLPVerifier(visual_dim, state_dim, action_dim, chunk_size, hidden_dim)
    if kind == "gru":
        return GRUVerifier(visual_dim, state_dim, action_dim, chunk_size, hidden_dim)
    if kind == "transformer":
        return TinyTransformerVerifier(visual_dim, state_dim, action_dim, chunk_size, hidden_dim)
    raise ValueError(f"Unknown model kind: {kind}")


@torch.inference_mode()
def benchmark(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.dtype == "fp16":
        dtype = torch.float16
    elif args.dtype == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    model = build_model(
        kind=args.kind,
        visual_dim=args.visual_dim,
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        chunk_size=args.chunk_size,
        hidden_dim=args.hidden_dim,
    ).to(device=device, dtype=dtype).eval()

    z = torch.randn(args.batch_size, args.visual_dim, device=device, dtype=dtype)
    state = torch.randn(args.batch_size, args.state_dim, device=device, dtype=dtype)
    actions = torch.randn(args.batch_size, args.chunk_size, args.action_dim, device=device, dtype=dtype)

    # Warmup
    for _ in range(args.warmup):
        _ = model(z, state, actions)

    if device == "cuda":
        torch.cuda.synchronize()

    times = []

    for _ in range(args.runs):
        if device == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            _ = model(z, state, actions)
            end.record()

            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        else:
            t0 = time.perf_counter()
            _ = model(z, state, actions)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

    times_t = torch.tensor(times)

    num_params = sum(p.numel() for p in model.parameters())

    print("==== JEPA-style verifier latency benchmark ====")
    print(f"device       : {device}")
    print(f"dtype        : {args.dtype}")
    print(f"model        : {args.kind}")
    print(f"params       : {num_params / 1e6:.3f} M")
    print(f"batch_size   : {args.batch_size}")
    print(f"visual_dim   : {args.visual_dim}")
    print(f"chunk_size   : {args.chunk_size}")
    print(f"hidden_dim   : {args.hidden_dim}")
    print("-----------------------------------------------")
    print(f"mean latency : {times_t.mean().item():.4f} ms")
    print(f"p50 latency  : {times_t.median().item():.4f} ms")
    print(f"p95 latency  : {times_t.quantile(0.95).item():.4f} ms")
    print(f"min latency  : {times_t.min().item():.4f} ms")
    print(f"max latency  : {times_t.max().item():.4f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", type=str, default="mlp", choices=["mlp", "gru", "transformer"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--visual-dim", type=int, default=1024)
    parser.add_argument("--state-dim", type=int, default=7)
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--chunk-size", type=int, default=12)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--runs", type=int, default=500)
    args = parser.parse_args()

    benchmark(args)