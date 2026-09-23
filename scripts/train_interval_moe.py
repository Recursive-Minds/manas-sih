"""
T6 - Fine-tune the Bayesian MoE speed model with the interval (distance) loss.

    L = L_phase55(per-sample, subsampled) + lambda * L_int(alpha-emulated distance over 30-75 s)

Safety rules (enforced in code):
  * starts from the canonical checkpoint but NEVER writes it
    (models/checkpoints/best_moe_velocity_model.pt) and never overwrites any
    existing file unless --overwrite is given;
  * train = Part 1, val = Part 2 of S-M / S-S1 / S-S2 only (Part 3 and S-S3a / S-S4 never loaded);
  * normalisation = the base checkpoint's mean/std.

Example (repo root):
    python scripts/train_interval_moe.py --lam 1.0 --epochs 8 --seed 42
Output: models/checkpoints/round1_interval_lam1.0_s42.pt + logs/round1/train_interval_lam1.0_s42.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath("."))
from sih.data.loader import GenericDataLoader
from sih.models.resnet1d import ResNet1DSpeedEstimator
from sih.models.tcn_attention import TCNAttentionVelocityModel
from sih.models.moe_fusion import BayesianMoEFusion
from sih.models.losses import phase55_balanced_loss
from sih.models.interval_dataset import IntervalSequenceSampler, windows_from_sequence
from sih.models.interval_loss import emulate_alpha, interval_distance_loss, interval_error_final

CANONICAL = "models/checkpoints/best_moe_velocity_model.pt"
TRAIN_TRIPS = ["S-S1", "S-S2", "S-M"]


def load_base(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    c = ckpt.get("in_channels", 12)
    res = ResNet1DSpeedEstimator(in_channels=c, base_channels=64)
    tcn = TCNAttentionVelocityModel(in_channels=c, base_channels=32, num_attention_heads=4)
    res.load_state_dict(ckpt["expert_resnet_state_dict"])
    tcn.load_state_dict(ckpt["expert_tcn_state_dict"])
    model = BayesianMoEFusion(res, tcn).to(device)
    mean = np.asarray(ckpt.get("norm_mean", ckpt.get("mean")), dtype=np.float32).reshape(1, -1)
    std = np.asarray(ckpt.get("norm_std", ckpt.get("std")), dtype=np.float32).reshape(1, -1)
    return model, res, tcn, mean, std, c, ckpt


def forward_seq(model, batch, device, amp):
    xs, xl = windows_from_sequence(batch["x"].to(device))
    with torch.amp.autocast("cuda", enabled=amp):
        vf, varf, diag = model(xs, xl)
    B, L = batch["v"].shape
    return vf.float().view(B, L), varf, diag


def evaluate(model, sampler, val_sets, device, amp, cal_n, chunk=4):
    model.eval()
    errs, sq, n, sum_p, sum_g = [], 0.0, 0, 0.0, 0.0
    with torch.no_grad():
        for H, seqs in val_sets.items():
            for k in range(0, len(seqs), chunk):
                batch = sampler.build(seqs[k:k + chunk], H)
                v_hat, _, _ = forward_seq(model, batch, device, amp)
                v_hat = torch.nan_to_num(v_hat, nan=0.0, posinf=35.0, neginf=0.0)
                v = batch["v"].to(device)
                alpha = emulate_alpha(v_hat[:, :cal_n], v[:, :cal_n])
                errs += interval_error_final(v_hat[:, cal_n:], v[:, cal_n:], alpha).cpu().tolist()
                d = (v_hat - v)
                sq += float((d ** 2).sum()); n += d.numel()
                sum_p += float(v_hat.sum()); sum_g += float(v.sum())
    e = np.asarray(errs)
    return {"int_err_median": float(np.median(e)) if len(e) else float("nan"),
            "int_err_p90": float(np.percentile(e, 90)) if len(e) else float("nan"),
            "rmse": float(np.sqrt(sq / max(n, 1))), "scale_ratio": sum_p / max(sum_g, 1e-6), "n_seq": int(len(e))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=CANONICAL)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--steps", type=int, default=150, help="optimizer steps per epoch")
    ap.add_argument("--batch", type=int, default=4, help="sequences per step (lower to 2 on CUDA OOM)")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cal-s", type=float, default=20.0)
    ap.add_argument("--horizons", type=float, nargs="+", default=[30.0, 45.0, 60.0, 75.0])
    ap.add_argument("--pointwise-stride", type=int, default=5, help="every k-th step feeds the per-sample loss")
    ap.add_argument("--max-rmse-increase", type=float, default=0.10, help="reject epochs whose val RMSE worsens more than this fraction")
    ap.add_argument("--out", default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="2 epochs x 3 steps, for a pipeline check only")
    args = ap.parse_args()

    tag = f"lam{args.lam}_s{args.seed}" + ("_smoke" if args.smoke else "")
    out = args.out or f"models/checkpoints/round1_interval_{tag}.pt"
    log_path = f"logs/round1/train_interval_{tag}.json"
    if os.path.abspath(out) == os.path.abspath(CANONICAL):
        sys.exit("Refusing to write the canonical checkpoint.")
    if os.path.exists(out) and not args.overwrite:
        sys.exit(f"{out} exists. Use --overwrite or --out.")
    if args.smoke:
        args.epochs, args.steps = 2, 3
    os.makedirs(os.path.dirname(out), exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    print(f"[T6] device={device} lambda={args.lam} base={args.base} -> {out}")

    model, res, tcn, mean, std, in_ch, base_ckpt = load_base(args.base, device)
    loader = GenericDataLoader()
    trips = [loader.load_file(os.path.join("data/raw/iovnbd_trips", f"{t}.csv")) for t in TRAIN_TRIPS]
    train_s = IntervalSequenceSampler.from_trips(trips, "train", mean, std, in_channels=in_ch, cal_s=args.cal_s)
    val_s = IntervalSequenceSampler.from_trips(trips, "val", mean, std, in_channels=in_ch, cal_s=args.cal_s)
    cal_n = train_s.cal_n
    val_sets = {H: val_s.fixed_set(H, n_per_trip=4 if args.smoke else 12) for H in args.horizons}
    print(f"[T6] val sequences: { {h: len(s) for h, s in val_sets.items()} }")

    base_metrics = evaluate(model, val_s, val_sets, device, amp, cal_n)
    print(f"[T6] epoch 0 (base)  {base_metrics}")
    history = [{"epoch": 0, **base_metrics}]

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs), eta_min=args.lr * 0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    best = {"score": base_metrics["int_err_median"], "epoch": 0}
    rmse_cap = base_metrics["rmse"] * (1.0 + args.max_rmse_increase)
    t0 = time.time()

    for ep in range(1, args.epochs + 1):
        model.train()
        l_pt_sum = l_int_sum = 0.0
        done = 0
        for _ in range(args.steps):
            H = float(rng.choice(args.horizons))
            batch = train_s.sample(args.batch, H, rng, augment=True)
            if batch is None:
                continue
            opt.zero_grad()
            v_hat, var_f, diag = forward_seq(model, batch, device, amp)
            v = batch["v"].to(device)
            B, L = v.shape
            sel = torch.arange(0, B * L, args.pointwise_stride, device=device)
            flat = lambda t: t.float().reshape(B * L, -1).index_select(0, sel)
            with torch.amp.autocast("cuda", enabled=amp):
                l_pt = phase55_balanced_loss(
                    v_fused=flat(v_hat), var_fused=flat(var_f),
                    v_res=flat(diag["v_resnet"]), var_res=flat(diag["var_resnet"]),
                    v_tcn=flat(diag["v_tcn"]), var_tcn=flat(diag["var_tcn"]),
                    v_gt=flat(v), a_lat=flat(batch["a_lat"].to(device)), w_yaw=flat(batch["w_yaw"].to(device)),
                    class_logits=diag["class_logits"].float().reshape(B * L, -1).index_select(0, sel),
                    motion_labels=batch["label"].to(device).reshape(-1).index_select(0, sel),
                )
            alpha = emulate_alpha(v_hat[:, :cal_n], v[:, :cal_n])
            l_int = interval_distance_loss(v_hat[:, cal_n:], v[:, cal_n:], alpha)
            loss = l_pt + args.lam * l_int
            if not torch.isfinite(loss):
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            scaler.step(opt)
            scaler.update()
            l_pt_sum += l_pt.item(); l_int_sum += l_int.item(); done += 1
        sched.step()
        m = evaluate(model, val_s, val_sets, device, amp, cal_n)
        ok = m["rmse"] <= rmse_cap
        is_best = ok and m["int_err_median"] < best["score"]
        row = {"epoch": ep, "train_pointwise": l_pt_sum / max(done, 1), "train_interval": l_int_sum / max(done, 1),
               **m, "rmse_guard_ok": ok, "best": is_best}
        history.append(row)
        print(f"[T6] epoch {ep}/{args.epochs} {row}")
        if is_best:
            best = {"score": m["int_err_median"], "epoch": ep}
            torch.save({
                "epoch": ep, "model_type": "moe_bayesian",
                "expert_resnet_state_dict": res.state_dict(), "expert_tcn_state_dict": tcn.state_dict(),
                "norm_mean": mean, "norm_std": std, "in_channels": in_ch, "short_len": 20, "long_len": 60,
                "val_rmse": m["rmse"], "speed_scale_ratio": m["scale_ratio"],
                "round1": {"lambda": args.lam, "base": args.base, "val": m, "seed": args.seed},
            }, out)
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump({"args": vars(args), "out": out, "best": best, "history": history,
                       "elapsed_s": time.time() - t0}, f, indent=2)

    if best["epoch"] == 0:
        print("[T6] no epoch beat the base checkpoint on validation; nothing saved.")
    else:
        print(f"[T6] best epoch {best['epoch']} val int_err_median {best['score']:.4f} "
              f"(base {base_metrics['int_err_median']:.4f}) -> {out}")
    print(f"[T6] log: {log_path}")


if __name__ == "__main__":
    main()
