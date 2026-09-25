# -*- coding: utf-8 -*-
"""One-frame debug: check per-class tp/fp/gt and label distribution."""
import os
import sys
from collections import OrderedDict, defaultdict

import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils
from opencood.utils import eval_utils_airv2x as eval_utils
from opencood.utils import common_utils
import numpy as np


def main():
    model_dir = sys.argv[1]
    epoch = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    device = torch.device("cuda:0")

    hypes = yaml_utils.load_yaml(None, __import__("argparse").Namespace(model_dir=model_dir))
    hypes["validate_dir"] = hypes["test_dir"]
    dataset = build_dataset(hypes, visualize=True, train=False)
    # first frame of first scenario
    idx0 = 0
    loader = DataLoader(
        Subset(dataset, [idx0]), batch_size=1, num_workers=0,
        collate_fn=dataset.collate_batch_test, shuffle=False,
    )
    batch = next(iter(loader))
    batch = train_utils.to_device(batch, device)

    model = train_utils.create_model(hypes).to(device)
    sd = torch.load(os.path.join(model_dir, f"net_epoch{epoch}.pth"), map_location=device)
    if "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    sd = {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    model.eval()

    with torch.no_grad():
        output_dict = OrderedDict()
        output_dict["ego"] = model(batch["ego"])
        det_b, det_s, det_l, _ = dataset.post_processor.post_process_airv2x(batch, output_dict)
        gt_b, gt_c, _ = dataset.post_processor.generate_gt_bbx_airv2x(batch)

    print("pred boxes:", None if det_b is None else det_b.shape)
    print("pred labels:", None if det_l is None else det_l.cpu().tolist())
    print("pred scores:", None if det_s is None else [round(x, 3) for x in det_s.cpu().tolist()])
    print("gt boxes:", gt_b.shape)
    print("gt labels:", gt_c)

    # class-agnostic IoU of each pred vs all GT
    if det_b is not None and len(det_b) > 0:
        det_np = common_utils.torch_tensor_to_numpy(det_b)
        gt_np = common_utils.torch_tensor_to_numpy(gt_b)
        det_polys = list(common_utils.convert_format(det_np))
        gt_polys = list(common_utils.convert_format(gt_np))
        for i, dp in enumerate(det_polys):
            ious = common_utils.compute_iou(dp, gt_polys)
            j = int(np.argmax(ious))
            print(f"pred[{i}] cls={det_l[i].item()} score={det_s[i].item():.3f} "
                  f"bestIoU={ious[j]:.3f} vs gt[{j}] cls={gt_c[j]}")

    # run the real accumulation for one frame, then inspect
    result_stat = defaultdict(dict)
    eval_utils.calculate_multiclass_tp_fp(
        det_b, det_s, det_l, gt_b, gt_c, iou_thresh=0.3, result_stat=result_stat
    )
    for cid, d in result_stat.items():
        print(f"class {cid} @0.3:", {k: v for k, v in d[0.3].items()})


if __name__ == "__main__":
    main()
