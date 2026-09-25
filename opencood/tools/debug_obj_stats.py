# -*- coding: utf-8 -*-
"""Quick check: objectness sigmoid distribution & detection counts on val samples."""
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils
from torch.utils.data import DataLoader


def main():
    model_dir = sys.argv[1]
    device = torch.device("cuda:0")
    hypes = yaml_utils.load_yaml(None, __import__("argparse").Namespace(model_dir=model_dir))
    dataset = build_dataset(hypes, visualize=False, train=False)
    loader = DataLoader(dataset, batch_size=1, num_workers=0,
                        collate_fn=dataset.collate_batch_test, shuffle=False)

    model = train_utils.create_model(hypes).to(device)
    sd = torch.load(os.path.join(model_dir, "net_epoch12.pth"), map_location="cpu")
    sd = sd.get("model_state_dict", sd)
    model.load_state_dict({k.replace("module.", "", 1): v for k, v in sd.items()})
    model.eval()

    n_pos_total, n_det = 0, 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= 5:
                break
            batch = train_utils.to_device(batch, device)
            out = model(batch["ego"])
            obj = torch.sigmoid(out["obj"])
            psm = out["psm"]
            print(f"[{i}] obj sigmoid: mean={obj.mean():.5f} max={obj.max():.5f} "
                  f"p999={obj.flatten().quantile(0.999):.5f} "
                  f">#0.2: {(obj > 0.2).sum().item()} | >#0.1: {(obj > 0.1).sum().item()}")
            # GT positives for reference
            pos = batch["ego"]["label_dict"]["pos_equal_one"]
            print(f"    GT pos anchors: {int(pos.sum())}")
            n_pos_total += int(pos.sum())

            # post-process to count final boxes
            pred_box3d, pred_score, pred_label = dataset.post_processor.post_process_airv2x(
                batch["ego"], out if isinstance(out, dict) else {"ego": out})
            n = 0 if pred_box3d is None else pred_box3d.shape[0]
            n_det += n
            print(f"    decoded boxes after NMS: {n}")
    print(f"SUMMARY: total GT pos anchors={n_pos_total}, total decoded boxes={n_det}")


if __name__ == "__main__":
    main()
