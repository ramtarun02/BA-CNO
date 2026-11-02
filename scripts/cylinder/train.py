"""Train MeshGraphNets / BA-CNO variants on CylinderFlow.

Designed for Mac prototyping and Colab/Kaggle runs: checkpoints every
``ckpt_every`` steps and resumes automatically from ``<output_dir>/last.pt``.

    python scripts/cylinder/train.py --config configs/cylinder/ba_cno_hard.yaml
    python scripts/cylinder/train.py --config configs/cylinder/mgn.yaml --override train.max_steps=200 data.max_trajectories=2
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, RandomSampler

from ba_cno.data.cylinder import CylinderFlowDataset, collate_graphs
from ba_cno.models import build_model
from ba_cno.normalization import Normalizer


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    for item in overrides:
        key, value = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(value)
    return cfg


def load_config(path: str, overrides: list[str]) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    return apply_overrides(cfg, overrides or [])


def fit_normalizers(dataset: CylinderFlowDataset, num_samples: int, center_target: bool, seed: int) -> dict:
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(len(dataset), (min(num_samples, len(dataset)),), generator=g).tolist()
    samples = [dataset[i] for i in idx]
    node = torch.cat([s["node_features"] for s in samples])
    edge = torch.cat([s["edge_features"] for s in samples])
    target = torch.cat([s["target"][s["loss_mask"]] for s in samples])
    norm = {
        "node": Normalizer(node.shape[-1]).fit(node),
        "edge": Normalizer(edge.shape[-1]).fit(edge),
        "target": Normalizer(target.shape[-1], center=center_target).fit(target),
    }
    # one-hot channels that never vary: keep them unscaled
    norm["node"].std[norm["node"].std < 1e-6] = 1.0
    return norm


def build(cfg: dict, node_in: int, edge_in: int):
    m = cfg["model"]
    kwargs = {k: v for k, v in m.items() if k != "name"}
    return build_model(m["name"], node_in=node_in, edge_in=edge_in, out_dim=3, **kwargs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.override)
    tr, data = cfg["train"], cfg["data"]

    torch.manual_seed(tr.get("seed", 0))
    device = pick_device(tr.get("device", "auto"))
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg))

    dataset = CylinderFlowDataset(data["root"], "train", data.get("max_trajectories"), noise_std=data.get("noise_std", 0.02))
    flux_decoder = cfg["model"].get("decoder") == "flux_divergence"
    sample = dataset[0]
    model = build(cfg, sample["node_features"].shape[-1], sample["edge_features"].shape[-1]).to(device)
    normalizers = fit_normalizers(dataset, tr.get("norm_samples", 200), center_target=not flux_decoder, seed=tr.get("seed", 0))
    normalizers = {k: v.to(device) for k, v in normalizers.items()}

    opt = torch.optim.Adam(model.parameters(), lr=tr.get("lr", 1e-4))
    max_steps = int(tr["max_steps"])
    # MeshGraphNets schedule: exponential decay towards lr_min
    decay = (tr.get("lr_min", 1e-6) / tr.get("lr", 1e-4)) ** (1.0 / max(max_steps, 1))
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=decay)

    step = 0
    history: list[dict] = []
    last = out_dir / "last.pt"
    if last.exists():
        ckpt = torch.load(last, map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        for k, v in ckpt["normalizers"].items():
            normalizers[k].load_state_dict(v)
        step, history = ckpt["step"], ckpt.get("history", [])
        print(f"Resumed from step {step}")

    sampler = RandomSampler(dataset, replacement=True, num_samples=10**9)
    loader = DataLoader(dataset, batch_size=tr.get("batch_size", 2), sampler=sampler,
                        collate_fn=collate_graphs, num_workers=tr.get("num_workers", 0))
    aux_w = float(tr.get("flux_penalty_weight", 0.0))
    p_w = float(tr.get("pressure_loss_weight", 0.0))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device} params={n_params:,} steps={max_steps}")

    def save(path: Path) -> None:
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "normalizers": {k: v.state_dict() for k, v in normalizers.items()},
                    "step": step, "history": history, "config": cfg}, path)

    model.train()
    t0 = time.time()
    running = 0.0
    for batch in loader:
        if step >= max_steps:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        batch["node_features"] = normalizers["node"](batch["node_features"])
        batch["edge_features"] = normalizers["edge"](batch["edge_features"])
        target = normalizers["target"](batch["target"])
        out = model(batch)
        err = (out["output"] - target).pow(2)
        mask = batch["loss_mask"]
        loss = err[mask, :2].sum(-1).mean()
        if p_w > 0:
            loss = loss + p_w * err[mask, 2].mean()
        if aux_w > 0 and "flux_antisymmetry" in out["aux_losses"]:
            loss = loss + aux_w * out["aux_losses"]["flux_antisymmetry"]
        if not math.isfinite(loss.item()):
            raise RuntimeError(f"Non-finite loss at step {step}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        step += 1
        running += loss.item()

        if step % tr.get("log_every", 100) == 0:
            rec = {"step": step, "loss": running / tr.get("log_every", 100), "lr": sched.get_last_lr()[0],
                   "sec": round(time.time() - t0, 1)}
            history.append(rec)
            print(json.dumps(rec))
            running = 0.0
        if step % tr.get("ckpt_every", 2000) == 0:
            save(last)

    save(last)
    save(out_dir / "final.pt")
    (out_dir / "history.json").write_text(json.dumps(history, indent=1))
    print(f"done: {step} steps in {time.time() - t0:.0f}s -> {out_dir}")


if __name__ == "__main__":
    main()
