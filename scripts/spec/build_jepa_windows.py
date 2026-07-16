from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--future-steps", type=int, default=4)
    parser.add_argument("--action-dim", type=int, default=32)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    files = sorted(glob.glob(str(raw_dir / "sample_*.npz")))

    if len(files) <= args.future_steps:
        raise RuntimeError(
            f"Not enough samples: {len(files)}, future_steps={args.future_steps}"
        )

    z_list = []
    state_list = []
    actions_list = []

    for p in files:
        d = np.load(p)

        z = d["z_t"].astype(np.float32)
        state = d["state_t"].astype(np.float32)
        actions = d["actions"].astype(np.float32)

        if z.ndim == 2:
            z = z[0]
        if state.ndim == 2:
            state = state[0]
        if actions.ndim == 3:
            actions = actions[0]

        actions = actions[:, : args.action_dim]

        z_list.append(z)
        state_list.append(state)
        actions_list.append(actions)

    z_arr = np.stack(z_list, axis=0)
    state_arr = np.stack(state_list, axis=0)
    actions_arr = np.stack(actions_list, axis=0)

    K = int(args.future_steps)

    z_t_out = []
    state_t_out = []
    actions_out = []
    z_future_out = []

    for i in range(0, len(files) - K):
        z_t_out.append(z_arr[i])
        state_t_out.append(state_arr[i])
        actions_out.append(actions_arr[i, :K, :])
        z_future_out.append(z_arr[i + 1 : i + 1 + K])

    z_t_out = np.stack(z_t_out, axis=0)
    state_t_out = np.stack(state_t_out, axis=0)
    actions_out = np.stack(actions_out, axis=0)
    z_future_out = np.stack(z_future_out, axis=0)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_path,
        z_t=z_t_out,
        state_t=state_t_out,
        actions=actions_out,
        z_future=z_future_out,
        raw_files=np.asarray(files),
    )

    print("saved:", out_path)
    print("num raw samples:", len(files))
    print("num windows:", z_t_out.shape[0])
    print("z_t:", z_t_out.shape)
    print("state_t:", state_t_out.shape)
    print("actions:", actions_out.shape)
    print("z_future:", z_future_out.shape)


if __name__ == "__main__":
    main()