# -*- coding: utf-8 -*-
"""Quick epoch-comparison evaluation on a small sample of the test set.

Loads several checkpoints (e.g. epoch 1 / 5 / 9) from one run directory and
compares detection quality on the first ``--frames-per-scenario`` frames of
each test scenario directory. Uses the same multiclass TP/FP + mAP machinery
as the full AirV2X evaluation, so numbers are comparable in scale (but
noisier due to the small sample).

Example:
    python opencood/tools/inference_quick_compare.py \
        --model_dir opencood/logs/.../cam_pred_depth_xxx \
        --epochs 1 5 9 --frames-per-scenario 5
"""

import argparse
import os
import sys
from collections import OrderedDict, defaultdict

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import inference_utils, train_utils
from opencood.utils import eval_utils_airv2x as eval_utils

# how the dataset orders scenarios -> one index block per scenario dir
import bisect


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="quick epoch compare")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, nargs="+", default=[1, 5, 9])
    parser.add_argument("--frames-per-scenario", type=int, default=5)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    return parser.parse_args()


def select_indices(dataset, frames_per_scenario: int):
    """Pick the first N frame indices of every scenario block."""
    len_record = dataset.len_record
    starts = [0] + [r for r in len_record[:-1]]
    idxs = []
    for s, e in zip(starts, len_record):
        idxs.extend(range(s, min(s + frames_per_scenario, e)))
    return idxs


def main():
    opt = parse_args()
    torch.cuda.set_device(opt.gpu_id)
    device = torch.device(f"cuda:{opt.gpu_id}")

    hypes = yaml_utils.load_yaml(None, argparse.Namespace(model_dir=opt.model_dir))
    hypes["validate_dir"] = hypes["test_dir"]

    print("Building test dataset ...")
    dataset = build_dataset(hypes, visualize=True, train=False)
    print(f"{len(dataset)} test samples, {len(dataset.len_record)} scenarios")

    subset_idx = select_indices(dataset, opt.frames_per_scenario)
    print(f"Sampling {len(subset_idx)} frames ({opt.frames_per_scenario}/scenario)")

    loader = DataLoader(
        Subset(dataset, subset_idx),
        batch_size=opt.batch_size,
        num_workers=2,
        collate_fn=dataset.collate_batch_test,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )

    # Cache all batches once on CPU; re-run every epoch's ckpt over them.
    print("Loading & caching sample batches ...")
    batches = []
    for batch in tqdm(loader):
        batches.append(batch)

    model = train_utils.create_model(hypes).to(device)
    model.eval()

    for epoch in opt.epochs:
        ckpt = os.path.join(opt.model_dir, f"net_epoch{epoch}.pth")
        if not os.path.exists(ckpt):
            print(f"[skip] {ckpt} not found")
            continue
        sd = torch.load(ckpt, map_location=device)
        if "model_state_dict" in sd:  # full training-state checkpoint
            sd = sd["model_state_dict"]
        sd = {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in sd.items()}
        model.load_state_dict(sd, strict=True)

        all_det_boxes, all_det_scores, all_det_labels = [], [], []
        all_gt_boxes, all_gt_labels = [], []
        n_gt_total = 0

        with torch.no_grad():
            for batch in batches:
                batch = train_utils.to_device(batch, device)
                # direct call: dataset.post_process returns 7 values but
                # inference_utils.inference_early_fusion_airv2x unpacks 6 (broken)
                output_dict = OrderedDict()
                output_dict["ego"] = model(batch["ego"])
                pred_box, pred_score, pred_labels, _ = dataset.post_processor.post_process_airv2x(
                    batch, output_dict
                )
                gt_box, gt_class, _ = dataset.post_processor.generate_gt_bbx_airv2x(batch)
                all_det_boxes.append(pred_box)
                all_det_scores.append(pred_score)
                all_det_labels.append(pred_labels)
                all_gt_boxes.append(gt_box)
                all_gt_labels.append(gt_class)
                if gt_box is not None:
                    n_gt_total += gt_box.shape[0]

        # per-frame accumulation: calculate_multiclass_tp_fp takes ONE sample
        result_stat = defaultdict(dict)
        for det_b, det_s, det_l, gt_b, gt_c in zip(
            all_det_boxes, all_det_scores, all_det_labels, all_gt_boxes, all_gt_labels
        ):
            for iou in (0.3, 0.5, 0.7):
                if det_b is not None and len(det_b) > 0:
                    eval_utils.calculate_multiclass_tp_fp(
                        det_b, det_s, det_l, gt_b, gt_c,
                        iou_thresh=iou, result_stat=result_stat,
                    )
                elif gt_b is not None:
                    # no predictions this frame: still count GT so the
                    # recall denominator stays correct
                    for cls_id in set(gt_c):
                        ent = result_stat.setdefault(cls_id, {}).setdefault(
                            iou, {"tp": [], "fp": [], "score": [], "gt": 0}
                        )
                        ent["gt"] += sum(1 for c in gt_c if c == cls_id)
        print(f"\n===== epoch {epoch} ({len(batches)} frames, {n_gt_total} GT boxes) =====")

        # class-agnostic AP (same protocol as historical eval_epochNone.yaml)
        aggr = {iou: {"tp": [], "fp": [], "score": [], "gt": 0} for iou in (0.3, 0.5, 0.7)}
        for det_b, det_s, gt_b in zip(all_det_boxes, all_det_scores, all_gt_boxes):
            for iou in (0.3, 0.5, 0.7):
                if gt_b is None or len(gt_b) == 0:
                    continue
                if det_b is None or len(det_b) == 0:
                    aggr[iou]["gt"] += gt_b.shape[0]
                    continue
                eval_utils.caluclate_tp_fp(det_b, det_s, gt_b, aggr, iou)
        for iou in (0.3, 0.5, 0.7):
            ap, _, _ = eval_utils.calculate_ap(aggr, iou, global_sort_detections=True)
            print(f"[class-agnostic] AP@{iou:.2f} = {ap:.4f}")

        # geometry diag: best IoU of each pred against GT (class-agnostic)
        from opencood.utils import common_utils
        best_ious = []
        for det_b, gt_b in zip(all_det_boxes, all_gt_boxes):
            if det_b is None or len(det_b) == 0 or gt_b is None or len(gt_b) == 0:
                continue
            det_np = common_utils.torch_tensor_to_numpy(det_b)
            gt_np = common_utils.torch_tensor_to_numpy(gt_b)
            det_polys = list(common_utils.convert_format(det_np))
            gt_polys = list(common_utils.convert_format(gt_np))
            frame_best = [
                max(common_utils.compute_iou(dp, gt_polys), default=0.0)
                for dp in det_polys
            ]
            best_ious.extend(frame_best)
        if best_ious:
            import numpy as _np
            print(f"[diag] preds={len(best_ious)} | best-IoU mean={_np.mean(best_ious):.3f} "
                  f">0.3: {sum(i > 0.3 for i in best_ious)} | >0.5: {sum(i > 0.5 for i in best_ious)}")

        for iou in (0.3, 0.5, 0.7):
            ap_per_class, mAP = eval_utils.compute_multiclass_ap_map(
                result_stat, iou_thresh=iou, global_sort_detections=True
            )
            detail = " | ".join(
                f"c{cid}: {ap:.3f}" for cid, ap in sorted(ap_per_class.items())
            )
            print(f"mAP@{iou:.1f} = {mAP:.4f}   [{detail}]")


if __name__ == "__main__":
    main()
