#!/usr/bin/env python3
"""Benchmark HEAL / STAMP inference latency for paper tables."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HEAL/STAMP latency benchmark")
    parser.add_argument("--heal_dir", type=str, required=True)
    parser.add_argument("--heal_epoch", type=int, default=19)
    parser.add_argument("--stamp_dir", type=str, required=True)
    parser.add_argument("--stamp_epoch", type=int, default=25)
    parser.add_argument("--fusion_method", type=str, default="intermediate")
    parser.add_argument("--num_frames", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--topk_slow", type=int, default=8)
    parser.add_argument("--out_json", type=str, default="")
    return parser.parse_args()


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _copy_numpy_contiguous(inputs: Any) -> Any:
    if isinstance(inputs, dict):
        return {k: _copy_numpy_contiguous(v) for k, v in inputs.items()}
    if isinstance(inputs, list):
        return [_copy_numpy_contiguous(v) for v in inputs]
    if isinstance(inputs, tuple):
        return tuple(_copy_numpy_contiguous(v) for v in inputs)
    if isinstance(inputs, np.ndarray):
        return np.ascontiguousarray(inputs.copy()) if not inputs.flags["C_CONTIGUOUS"] or any(s < 0 for s in inputs.strides) else inputs.copy()
    return inputs


def _agent_count(batch_data: Dict[str, Any]) -> int:
    ego = batch_data["ego"]
    if "record_len" in ego:
        rl = ego["record_len"]
        if torch.is_tensor(rl):
            return int(rl.sum().item())
        if isinstance(rl, np.ndarray):
            return int(rl.sum())
        if isinstance(rl, (list, tuple)):
            return int(sum(rl))
    if "anchor_box" in ego:
        return 1
    return 1


def _summarize(xs: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(xs, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p90_ms": float(np.percentile(arr, 90)),
        "p95_ms": float(np.percentile(arr, 95)),
        "max_ms": float(arr.max()),
    }


def load_model_and_dataset(model_dir: str, epoch: int, fusion_method: str) -> Tuple[Dict[str, Any], Any, torch.nn.Module, int]:
    opt = argparse.Namespace(
        model_dir=model_dir,
        config_file="config.yaml",
        fusion_method=fusion_method,
        eval_epoch=epoch,
        eval_best_epoch=False,
    )
    hypes = yaml_utils.load_yaml(None, opt)
    hypes["validate_dir"] = hypes.get("validate_dir", hypes.get("test_dir"))
    dataset = build_dataset(hypes, visualize=False, train=False)
    model = train_utils.create_model(hypes)
    epoch_id, model = train_utils.load_model(model_dir, model, epoch, start_from_best=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return hypes, dataset, model, epoch_id


def run_once_timings(model: torch.nn.Module, dataset: Any, batch_cpu: Dict[str, Any], device: torch.device) -> Dict[str, float]:
    _sync()
    t0 = time.perf_counter()
    batch = train_utils.to_device(_copy_numpy_contiguous(batch_cpu), device)
    _sync()
    t1 = time.perf_counter()
    with torch.no_grad():
        out = model(batch["ego"])
    _sync()
    t2 = time.perf_counter()
    output_dict = {"ego": out}
    _ = dataset.post_process(batch, output_dict)
    _sync()
    t3 = time.perf_counter()
    return {
        "h2d_ms": (t1 - t0) * 1000.0,
        "forward_ms": (t2 - t1) * 1000.0,
        "post_ms": (t3 - t2) * 1000.0,
        "fwd_post_ms": (t3 - t1) * 1000.0,
        "e2e_h2d_ms": (t3 - t0) * 1000.0,
    }


def benchmark_model(name: str, model_dir: str, epoch: int, fusion_method: str, num_frames: int, warmup: int, repeats: int, topk_slow: int) -> Dict[str, Any]:
    print(f"\n===== {name} =====")
    hypes, dataset, model, epoch_id = load_model_and_dataset(model_dir, epoch, fusion_method)
    device = next(model.parameters()).device
    print(f"loaded epoch={epoch_id}, device={device}, dataset_len={len(dataset)}")

    picked = list(range(min(num_frames, len(dataset))))
    batches: List[Dict[str, Any]] = []
    agent_counts: List[int] = []
    for idx in picked:
        batch = dataset.collate_batch_test([dataset[idx]])
        batches.append(batch)
        agent_counts.append(_agent_count(batch))
    print(f"using {len(picked)} frames")
    print(
        "agent_count: "
        f"min={min(agent_counts)} median={int(np.median(agent_counts))} "
        f"max={max(agent_counts)} mean={np.mean(agent_counts):.2f}"
    )

    for _ in range(warmup):
        _ = run_once_timings(model, dataset, batches[0], device)

    per_frame: List[Dict[str, float]] = []
    for bi, batch in enumerate(batches):
        reps = [run_once_timings(model, dataset, batch, device) for _ in range(repeats)]
        frame_stat: Dict[str, float] = {
            "frame_idx": float(picked[bi]),
            "agent_count": float(agent_counts[bi]),
        }
        for key in reps[0].keys():
            vals = [r[key] for r in reps]
            frame_stat[key] = float(np.median(vals))
        per_frame.append(frame_stat)
        print(
            f"[{name}] frame={picked[bi]:05d} agents={agent_counts[bi]:02d} "
            f"fwd={frame_stat['forward_ms']:.1f}ms "
            f"fwd+post={frame_stat['fwd_post_ms']:.1f}ms "
            f"e2e+h2d={frame_stat['e2e_h2d_ms']:.1f}ms"
        )

    def collect(key: str) -> List[float]:
        return [float(f[key]) for f in per_frame]

    e2e = collect("e2e_h2d_ms")
    order = np.argsort(np.asarray(e2e))[::-1].copy()
    k = min(topk_slow, len(order))
    slow = [e2e[int(i)] for i in order[:k]]

    return {
        "name": name,
        "model_dir": model_dir,
        "epoch": int(epoch_id),
        "forward_ms": _summarize(collect("forward_ms")),
        "fwd_post_ms": _summarize(collect("fwd_post_ms")),
        "e2e_h2d_ms": _summarize(e2e),
        "cherry_pick": {
            "metric": "mean_of_topk_slowest_e2e_h2d_ms",
            "topk": k,
            "value_ms": float(np.mean(slow)),
            "frame_indices": [int(per_frame[int(i)]["frame_idx"]) for i in order[:k]],
            "values_ms": [float(v) for v in slow],
        },
        "per_frame": per_frame,
    }


def print_table(results: List[Dict[str, Any]]) -> None:
    print("\n========== Latency summary (ms) ==========")
    print(f"{'model':<8} {'fwd_mean':>9} {'fwd_p95':>9} {'e2e_mean':>9} {'e2e_p95':>9} {'cherry':>9}")
    for r in results:
        print(
            f"{r['name']:<8} "
            f"{r['forward_ms']['mean_ms']:9.1f} "
            f"{r['forward_ms']['p95_ms']:9.1f} "
            f"{r['e2e_h2d_ms']['mean_ms']:9.1f} "
            f"{r['e2e_h2d_ms']['p95_ms']:9.1f} "
            f"{r['cherry_pick']['value_ms']:9.1f}"
        )
    print("==========================================")


def main() -> None:
    args = parse_args()
    results = [
        benchmark_model("HEAL", args.heal_dir, args.heal_epoch, args.fusion_method, args.num_frames, args.warmup, args.repeats, args.topk_slow),
        benchmark_model("STAMP", args.stamp_dir, args.stamp_epoch, args.fusion_method, args.num_frames, args.warmup, args.repeats, args.topk_slow),
    ]
    print_table(results)
    out = {"args": vars(args), "results": results}
    out_path = args.out_json or os.path.join(args.heal_dir, "latency_heal_stamp.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
