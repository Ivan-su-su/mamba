# -*- coding: utf-8 -*-
# Author: OpenPCDet, Runsheng Xu <rxx3386@ucla.edu>
# Modifier: Yuheng Wu <yuhengwu@kaist.ac.kr>, Xiangbo Gao <xiangbogaobarry@gmail.com>
# License: TDG-Attribution-NonCommercial-NoDistrib


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedSmoothL1Loss(nn.Module):
    """
    Code-wise Weighted Smooth L1 Loss modified based on fvcore.nn.smooth_l1_loss
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/smooth_l1_loss.py
                  | 0.5 * x ** 2 / beta   if abs(x) < beta
    smoothl1(x) = |
                  | abs(x) - 0.5 * beta   otherwise,
    where x = input - target.
    """

    def __init__(self, beta: float = 1.0 / 9.0, code_weights: list = None):
        """
        Args:
            beta: Scalar float.
                L1 to L2 change point.
                For beta values < 1e-5, L1 loss is computed.
            code_weights: (#codes) float list if not None.
                Code-wise weights.
        """
        super(WeightedSmoothL1Loss, self).__init__()
        self.beta = beta
        if code_weights is not None:
            self.code_weights = np.array(code_weights, dtype=np.float32)
            self.code_weights = torch.from_numpy(self.code_weights).cuda()

    @staticmethod
    def smooth_l1_loss(diff, beta):
        if beta < 1e-5:
            loss = torch.abs(diff)
        else:
            n = torch.abs(diff)
            loss = torch.where(n < beta, 0.5 * n**2 / beta, n - 0.5 * beta)

        return loss

    def forward(
        self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor = None
    ):
        """
        Args:
            input: (B, #anchors, #codes) float tensor.
                Ecoded predicted locations of objects.
            target: (B, #anchors, #codes) float tensor.
                Regression targets.
            weights: (B, #anchors) float tensor if not None.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted smooth l1 loss without reduction.
        """
        target = torch.where(torch.isnan(target), input, target)  # ignore nan targets

        diff = input - target
        loss = self.smooth_l1_loss(diff, self.beta)

        # anchor-wise weighting
        if weights is not None:
            assert (
                weights.shape[0] == loss.shape[0] and weights.shape[1] == loss.shape[1]
            )
            loss = loss * weights.unsqueeze(-1)

        return loss


class PointPillarLossMultiClass(nn.Module):
    def __init__(self, args):
        super(PointPillarLossMultiClass, self).__init__()
        self.reg_loss_func = WeightedSmoothL1Loss()
        self.alpha = 0.25
        self.gamma = 2.0

        self.cls_weight = args["cls_weight"]
        self.reg_coe = args["reg"]
        self.obj_weight = args.get("obj_weight", 1.0)
        self.iou_weight = args.get("iou_weight", 0.0)  # 0.0 means disabled
        self.recall_weight = args.get("recall_weight", 0.0)  # 0.0 means disabled, for encouraging more detections
        self.flow_weight = args["flow_weight"] if "flow_weight" in args else 1.0
        self.loss_dict = {}
        self.use_dir = False
        self.cls_num = args["num_class"]
        gate_fg_cfg = args.get("GATE_FOREGROUND_AUX_LOSS", {}) or {}
        self.gate_aux_enabled = bool(gate_fg_cfg.get("ENABLED", True))
        if self.gate_aux_enabled:
            self.gate_loss_weight = float(gate_fg_cfg.get("LOSS_WEIGHT", 1.0))
            self.gate_warmup_epochs = int(gate_fg_cfg.get("WARMUP_EPOCHS", 3))
            self.dilation_kernel_size = int(gate_fg_cfg.get("DILATION_KERNEL_SIZE", 3))
            self.use_soft_target = bool(gate_fg_cfg.get("USE_SOFT_TARGET", True))
            self.soft_kernel_size = int(gate_fg_cfg.get("SOFT_TARGET_KERNEL_SIZE", 5))
            self.soft_sigma = float(gate_fg_cfg.get("SOFT_TARGET_SIGMA", 1.0))
            self.loss_type = str(gate_fg_cfg.get("LOSS_TYPE", "focal_bce")).lower()
            # self.gate_foreground_alpha = float(gate_fg_cfg.get("ALPHA", 0.25))
            self.aux_gamma = float(gate_fg_cfg.get("GAMMA", 2.0))
            self.gate_pos_weight = float(gate_fg_cfg.get("POS_WEIGHT", 2.0))
            self.gate_neg_visible_weight = float(gate_fg_cfg.get("NEG_VISIBLE_WEIGHT", 0.02))
            self.gate_neg_invisible_weight = float(gate_fg_cfg.get("NEG_INVISIBLE_WEIGHT", 0.05))
            # self.obj_gamma = args["obj_gamma"]
            # self.obj_alpha = args["obj_alpha"]
            # self.neg_weight = 0.2
        self.gate_debug_step = 0

        need_aux_cfg = args.get("NEED_FOREGROUND_AUX_LOSS", {}) or {}
        self.need_aux_enabled = bool(need_aux_cfg.get("ENABLED", False))
        if self.need_aux_enabled:
            self.need_loss_weight = float(need_aux_cfg.get("LOSS_WEIGHT", 0.5))
            self.need_aux_warmup_epochs = int(need_aux_cfg.get("WARMUP_EPOCHS", 3))
            self.dilation_kernel_size = int(need_aux_cfg.get("DILATION_KERNEL_SIZE", 3))
            self.use_soft_target = bool(need_aux_cfg.get("USE_SOFT_TARGET", True))
            self.soft_kernel_size = int(need_aux_cfg.get("SOFT_TARGET_KERNEL_SIZE", 5))
            self.soft_sigma = float(need_aux_cfg.get("SOFT_TARGET_SIGMA", 1.0))
            self.loss_type = str(need_aux_cfg.get("LOSS_TYPE", "focal_bce")).lower()
            self.aux_gamma = float(need_aux_cfg.get("GAMMA", 2.0))
            self.need_foreground_pos_weight = float(need_aux_cfg.get("POS_WEIGHT", 2.0))
            self.need_foreground_neg_rsu_invisible_weight = float(need_aux_cfg.get("NEG_RSU_INVISIBLE_WEIGHT", 0.02))
            self.need_foreground_neg_drone_invisible_weight = float(need_aux_cfg.get("NEG_DRONE_INVISIBLE_WEIGHT", 0.02))
        self.need_debug_step = 0


    def _gaussian_blur_mask(self, mask, kernel_size, sigma) -> torch.Tensor:
        """Apply depthwise Gaussian smoothing to a `[B, 1, H, W]` mask."""
        if kernel_size <= 2:
            return mask
        sigma = max(float(sigma), 1e-6)
        radius = kernel_size // 2
        coords = torch.arange(
            -radius,
            radius + 1,
            device=mask.device,
            dtype=mask.dtype,
        )
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        kernel = torch.exp(-(xx.pow(2) + yy.pow(2)) / (2.0 * sigma * sigma))
        kernel = kernel / torch.clamp(kernel.max(), min=1e-6)
        kernel = kernel.view(1, 1, kernel_size, kernel_size)
        soft_mask = F.conv2d(mask, kernel, padding=radius)
        return soft_mask.clamp(0.0, 1.0)

    def _build_target_from_pos_equal_one(self, pos_equal_one, target_shape) -> torch.Tensor:
        """Build a dense BEV foreground target from anchor positives.

        Anchor positives are very sparse, so we first collapse anchors with
        `amax`, then optionally dilate and smooth the BEV mask. This target is
        only a warm-up / auxiliary signal for gate foreground awareness; it is
        not the complete semantic target for final gate selection.
        """
        pos_bev = pos_equal_one.float().amax(dim=-1, keepdim=True)
        pos_bev = pos_bev.permute(0, 3, 1, 2).contiguous()
        pos_bev = pos_bev.clamp(0.0, 1.0)

        dilation_kernel = self.dilation_kernel_size
        
        if dilation_kernel > 2:
            pos_bev = F.max_pool2d(
                pos_bev,
                kernel_size=dilation_kernel,
                stride=1,
                padding=dilation_kernel // 2,
            )
            pos_bev = pos_bev.clamp(0.0, 1.0)

        if self.use_soft_target:
            hard_pos = pos_bev
            pos_bev = self._gaussian_blur_mask(
                pos_bev,
                self.soft_kernel_size,
                self.soft_sigma,
            )
            pos_bev = torch.maximum(pos_bev, hard_pos).clamp(0.0, 1.0)

        target_size = tuple(target_shape[-2:])
        if tuple(pos_bev.shape[-2:]) != target_size:
            if self.use_soft_target:
                pos_bev = F.interpolate(
                    pos_bev,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
            else:
                pos_bev = F.interpolate(
                    pos_bev,
                    size=target_size,
                    mode="nearest",
                )
        return pos_bev.clamp(0.0, 1.0)

    def _focal_bce_prob_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        alpha: float = 0.25,
        gamma: float = 2.0,
        pos_weight: float = 1.0,
    ) -> torch.Tensor:
        """Focal BCE for probability inputs such as sigmoid gate maps."""
        pred = pred.clamp(1e-4, 1.0 - 1e-4)
        target = target.float()
        bce = F.binary_cross_entropy(pred, target, reduction="none")
        pt = pred * target + (1.0 - pred) * (1.0 - target)
        focal_weight = (1.0 - pt).pow(gamma)
        alpha_weight = target * alpha + (1.0 - target) * (1.0 - alpha)
        focal_weight = focal_weight * alpha_weight
        if pos_weight is not None and pos_weight != 1.0:
            class_weight = target * float(pos_weight) + (1.0 - target)
            focal_weight = focal_weight * class_weight
        return (focal_weight * bce).mean()

    def _build_visibility_mask_from_extent(self, visibility_extent, pos_equal_one, gate_shape, device, dtype):
        """Build a source visibility mask from label-map rectangle extents."""
        if visibility_extent is None:
            return None
        extent = visibility_extent.to(device=device)
        batch_size = int(gate_shape[0])
        height, width = int(pos_equal_one.shape[1]), int(pos_equal_one.shape[2])
        mask = torch.zeros(
            (batch_size, 1, height, width),
            device=device,
            dtype=dtype,
        )
        extent_long = extent.long()
        valid = extent[:, 4] > 0
        for batch_idx in range(batch_size):
            if not bool(valid[batch_idx].item()):
                continue
            row_min = int(torch.clamp(extent_long[batch_idx, 0], 0, height - 1).item())
            row_max = int(torch.clamp(extent_long[batch_idx, 1], 0, height - 1).item())
            col_min = int(torch.clamp(extent_long[batch_idx, 2], 0, width - 1).item())
            col_max = int(torch.clamp(extent_long[batch_idx, 3], 0, width - 1).item())
            if row_max < row_min or col_max < col_min:
                continue
            mask[batch_idx, :, row_min : row_max + 1, col_min : col_max + 1] = 1.0

        if tuple(mask.shape[-2:]) != tuple(gate_shape[-2:]):
            mask = F.interpolate(mask, size=gate_shape[-2:], mode="nearest")
        return mask

    def _positive_prob_loss(self, pred, positive_weight):
        """Positive-only auxiliary loss that raises gate on visible foreground."""
        weight_sum = positive_weight.sum()
        if float(weight_sum.detach().item()) <= 0.0:
            return None
        pred = pred.clamp(1e-4, 1.0 - 1e-4)
        pos_bce = -torch.log(pred)
        if self.loss_type == "bce":
            loss_map = pos_bce
        else:
            focal_weight = (1.0 - pred).pow(self.aux_gamma)
            loss_map = pos_bce * focal_weight
        return (loss_map * positive_weight).sum() / torch.clamp(weight_sum, min=1.0)

    def _negative_prob_loss(self, pred, negative_mask):
        """Negative auxiliary loss that lowers gate on masked regions."""
        weight = negative_mask.float().detach()
        weight_sum = weight.sum()
        if float(weight_sum.detach().item()) <= 0.0:
            return None
        pred = pred.clamp(1e-4, 1.0 - 1e-4)
        neg_bce = -torch.log(1.0 - pred)
        if self.loss_type == "bce":
            loss_map = neg_bce
        else:
            focal_weight = pred.pow(self.aux_gamma)
            loss_map = neg_bce * focal_weight
        return (loss_map * weight).sum() / torch.clamp(weight_sum, min=1.0)

    def _compute_need_foreground_aux_loss(self, fusion_aux_outputs, pos_equal_one, target_dict, current_epoch):
        """Compute optional foreground auxiliary loss for temporal need map."""
        if not isinstance(fusion_aux_outputs, dict):
            return None

        need_map = fusion_aux_outputs.get("need_map", None)
        if not isinstance(need_map, torch.Tensor):
            return None
        if not need_map.requires_grad:
            return None

        # 这是总体的weight
        effective_weight = self.need_loss_weight * max(0.0, 1.0 - float(current_epoch) / float(self.need_aux_warmup_epochs))
        if effective_weight <= 0.0:
            return need_map.new_zeros(())

        need = need_map
        need_target = self._build_target_from_pos_equal_one(
            pos_equal_one.to(device=need.device, dtype=need.dtype),
            need.shape,
        ).detach()

        positive_weight = need_target.detach()
        non_fg = (need_target <= 0.0).detach()

        pos_loss = self._positive_prob_loss(
            pred=need,
            positive_weight=positive_weight,
        )
        if pos_loss is None:
            pos_loss = need.new_zeros(())

        source_visibility_masks = {
            "rsu": target_dict.get("gate_visibility_extent_rsu", None),
            "drone": target_dict.get("gate_visibility_extent_drone", None),
        }
        invisible_neg_masks = {}
        invisible_neg_losses = {}
        for source_name, visibility_extent in source_visibility_masks.items():
            visibility = self._build_visibility_mask_from_extent(visibility_extent, pos_equal_one, need.shape, need.device, need.dtype)
            if visibility is None:
                continue

            visibility = visibility.to(device=need.device, dtype=need.dtype).detach()
            invisible_neg_mask = (non_fg & (visibility <= 0.0)).detach()
            invisible_neg_loss = self._negative_prob_loss(
                pred=need,
                negative_mask=invisible_neg_mask,
            )
            if invisible_neg_loss is None:
                invisible_neg_loss = need.new_zeros(())

            invisible_neg_masks[source_name] = invisible_neg_mask
            invisible_neg_losses[source_name] = invisible_neg_loss

        rsu_neg_loss = invisible_neg_losses.get("rsu", need.new_zeros(()))
        drone_neg_loss = invisible_neg_losses.get("drone", need.new_zeros(()))

        loss_need_aux = (
            self.need_foreground_pos_weight * pos_loss
            + self.need_foreground_neg_rsu_invisible_weight * rsu_neg_loss
            + self.need_foreground_neg_drone_invisible_weight * drone_neg_loss
        )

        if self.need_debug_step % 100 == 0:
            rsu_neg_count = 0
            if isinstance(invisible_neg_masks.get("rsu", None), torch.Tensor):
                rsu_neg_count = int(invisible_neg_masks["rsu"].sum().detach().item())
            drone_neg_count = 0
            if isinstance(invisible_neg_masks.get("drone", None), torch.Tensor):
                drone_neg_count = int(invisible_neg_masks["drone"].sum().detach().item())
            print(
                f"[NeedAux] "
                f"need_map mean={need.mean().item():.6f} "
                f"positive_weight.sum()={positive_weight.sum().item():.2f} "
                f"rsu_invisible_neg_count={rsu_neg_count} "
                f"drone_invisible_neg_count={drone_neg_count} "
                f"pos_loss={pos_loss.item():.6f} "
                f"rsu_neg_loss={rsu_neg_loss.item():.6f} "
                f"drone_neg_loss={drone_neg_loss.item():.6f} "
            )
        self.need_debug_step += 1

        return loss_need_aux * effective_weight

    def _compute_gate_foreground_aux_loss(self, fusion_gate_outputs, pos_equal_one, current_epoch, gate_visibility_masks=None):
        """Compute optional foreground auxiliary loss for RSU/Drone gates."""

        source_gates = {
            "rsu": fusion_gate_outputs.get("gate_rsu", None),
            "drone": fusion_gate_outputs.get("gate_drone", None),
        }
        available_gates = [
            gate for gate in source_gates.values() if isinstance(gate, torch.Tensor)
        ]
        if not available_gates:
            return None
        if not any(gate.requires_grad for gate in available_gates):
            return None

        # 这是总体的weight，对于三个区域是相同的
        effective_weight = self.gate_loss_weight * max(0.0, 1.0 - float(current_epoch) / float(self.gate_warmup_epochs))
        if effective_weight <= 0.0:
            return available_gates[0].new_zeros(())

        reference_gate = available_gates[0]
        gate_target = self._build_target_from_pos_equal_one(
            pos_equal_one.to(device=reference_gate.device, dtype=reference_gate.dtype),
            reference_gate.shape,
        ).detach()

        loss_items = []
        gate_visibility_masks = gate_visibility_masks or {}
        for source_name, gate in source_gates.items():
            if not isinstance(gate, torch.Tensor):
                continue
            visibility = self._build_visibility_mask_from_extent(
                gate_visibility_masks.get(source_name),
                pos_equal_one,
                gate.shape,
                gate.device,
                gate.dtype,
            )
            if visibility is None:
                continue
            target = gate_target.to(device=gate.device, dtype=gate.dtype).detach()
            visibility = visibility.to(device=gate.device, dtype=gate.dtype).detach()
            positive_weight = (target * visibility).detach()
            visible_neg_mask = ((visibility > 0.0) & (target <= 0.0)).detach()
            invisible_neg_mask = (visibility <= 0.0).detach()

            pos_loss = self._positive_prob_loss(
                pred=gate,
                positive_weight=positive_weight,
            )
            visible_neg_loss = self._negative_prob_loss(
                pred=gate,
                negative_mask=visible_neg_mask,
            )
            invisible_neg_loss = self._negative_prob_loss(
                pred=gate,
                negative_mask=invisible_neg_mask,
            )

            if pos_loss is None:
                pos_loss = gate.new_zeros(())
            if visible_neg_loss is None:
                visible_neg_loss = gate.new_zeros(())
            if invisible_neg_loss is None:
                invisible_neg_loss = gate.new_zeros(())

            loss_item = (
                self.gate_pos_weight * pos_loss
                + self.gate_neg_visible_weight * visible_neg_loss
                + self.gate_neg_invisible_weight * invisible_neg_loss
            )
            loss_items.append(loss_item)

            if self.gate_debug_step % 5 == 0:
                print(
                    f"[gate_fg_aux][{source_name}] "
                    f"pos_loss={self.gate_pos_weight*pos_loss.item():.6f} "
                    f"visible_neg_loss={self.gate_neg_visible_weight*visible_neg_loss.item():.6f} "
                    f"invisible_neg_loss={self.gate_neg_invisible_weight*invisible_neg_loss.item():.6f} "
                    f"loss_item={loss_item.item():.6f}"
                )

        if not loss_items:
            return reference_gate.new_zeros(())

        loss_gate_aux = sum(loss_items) / len(loss_items)
        return loss_gate_aux * effective_weight

    def _debug_gate_grad_by_region(self, gate, target, det_loss, aux_loss, name="gate"):
        if gate is None or target is None:
            return
        if not isinstance(gate, torch.Tensor) or not gate.requires_grad:
            return
        if not isinstance(target, torch.Tensor):
            return

        target = target.detach().to(device=gate.device, dtype=gate.dtype)
        if target.shape[-2:] != gate.shape[-2:]:
            target = F.interpolate(
                target,
                size=gate.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        pos = target > 0.5
        neg = target <= 0.0

        def safe_grad_stats(grad, mask):
            if grad is None or not mask.any():
                return (-1.0, -1.0, -1.0)
            g = grad.detach().float()
            vals = g[mask]
            return (
                vals.mean().item(),
                vals.abs().mean().item(),
                (vals > 0).float().mean().item(),
            )

        try:
            det_grad = torch.autograd.grad(
                det_loss,
                gate,
                retain_graph=True,
                allow_unused=True,
            )[0]
        except Exception as e:
            print(f"[GateGrad][{name}] det_grad failed: {e}")
            det_grad = None

        try:
            aux_grad = None
            if isinstance(aux_loss, torch.Tensor) and aux_loss.requires_grad:
                aux_grad = torch.autograd.grad(
                    aux_loss,
                    gate,
                    retain_graph=True,
                    allow_unused=True,
                )[0]
        except Exception as e:
            print(f"[GateGrad][{name}] aux_grad failed: {e}")
            aux_grad = None

        det_pos_mean, det_pos_abs, det_pos_gt0 = safe_grad_stats(det_grad, pos)
        det_neg_mean, det_neg_abs, det_neg_gt0 = safe_grad_stats(det_grad, neg)

        aux_pos_mean, aux_pos_abs, aux_pos_gt0 = safe_grad_stats(aux_grad, pos)
        aux_neg_mean, aux_neg_abs, aux_neg_gt0 = safe_grad_stats(aux_grad, neg)

        print("=" * 80)
        print(f"[GateGrad][{name}]")
        print(
            "Note: grad > 0 means optimizer step tends to LOWER gate; "
            "grad < 0 means optimizer step tends to RAISE gate."
        )
        print(
            f"det_grad_pos_mean={det_pos_mean:.8f} "
            f"det_grad_pos_abs={det_pos_abs:.8f} "
            f"det_grad_pos_ratio_gt0={det_pos_gt0:.6f}"
        )
        print(
            f"det_grad_neg_mean={det_neg_mean:.8f} "
            f"det_grad_neg_abs={det_neg_abs:.8f} "
            f"det_grad_neg_ratio_gt0={det_neg_gt0:.6f}"
        )
        print(
            f"aux_grad_pos_mean={aux_pos_mean:.8f} "
            f"aux_grad_pos_abs={aux_pos_abs:.8f} "
            f"aux_grad_pos_ratio_gt0={aux_pos_gt0:.6f}"
        )
        print(
            f"aux_grad_neg_mean={aux_neg_mean:.8f} "
            f"aux_grad_neg_abs={aux_neg_abs:.8f} "
            f"aux_grad_neg_ratio_gt0={aux_neg_gt0:.6f}"
        )
        print("=" * 80)

    def forward(self, output_dict, target_dict, prefix=""):
        """
        Parameters
        ----------
        output_dict : dict
        target_dict : dict

        cls_label -> one_hot label
        """
        rm = output_dict["rm{}".format(prefix)]  # [B, #anchor*7, 50, 176]
        psm = output_dict["psm{}".format(prefix)]  # [B, #anchor*#class, 50, 176]
        obj = output_dict["obj{}".format(prefix)]  # [B, #anchor, 50, 176]
        targets = target_dict["targets"]
        
        cls_preds = psm.permute(0, 2, 3, 1).contiguous()  # N, C, H, W -> N, H, W, C
        obj_preds = obj.permute(0, 2, 3, 1).contiguous()  # (B, H, W, A)
        box_cls_labels = target_dict["pos_equal_one"]  # [B, 50, 176, 2]
        box_cls_labels = box_cls_labels.view(psm.shape[0], -1).contiguous()

        positives = box_cls_labels > 0
        negatives = box_cls_labels == 0
        negative_cls_weights = negatives * 1.0
        cls_weights = (negative_cls_weights + 1.0 * positives).float()
        reg_weights = positives.float()
        # added
        pos_mask = target_dict["pos_equal_one"]  # (B, H, W, A)
        neg_mask = target_dict["neg_equal_one"]  # (B, H, W, A)

        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)

        cls_labels = target_dict["class_ids"]  # [B, H, W, A]
        cls_targets = cls_labels
        one_hot_targets = torch.zeros(
            *list(cls_targets.shape),
            self.cls_num,
            dtype=cls_preds.dtype,
            device=cls_targets.device,
        )
        one_hot_targets.scatter_(-1, cls_targets.unsqueeze(dim=-1).long(), 1.0)
        cls_labels = one_hot_targets.view(
            cls_targets.shape[0], cls_targets.shape[1], cls_targets.shape[2], -1
        )
        
        cls_loss_src = self.cls_loss_func(
            cls_preds, cls_labels, weights=cls_weights
        )  # [N, M]
        
        cls_loss = cls_loss_src.sum() / psm.shape[0]
        conf_loss = cls_loss * self.cls_weight

        # regression
        rm = rm.permute(0, 2, 3, 1).contiguous()
        rm = rm.view(rm.size(0), -1, 7)
        targets = targets.view(targets.size(0), -1, 7)
        box_preds_sin, reg_targets_sin = self.add_sin_difference(rm, targets)
        loc_loss_src = self.reg_loss_func(
            box_preds_sin, reg_targets_sin, weights=reg_weights
        )

        reg_loss = loc_loss_src.sum() / rm.shape[0]
        reg_loss *= self.reg_coe

        # objectness loss using Focal Loss (reuse obj_loss_func)
        obj_preds_expanded = obj_preds.unsqueeze(-1)  # [B, H, W, A] -> [B, H, W, A, 1]
        pos_mask_expanded = pos_mask.unsqueeze(-1).float()  # [B, H, W, A] -> [B, H, W, A, 1]
        obj_loss_src = self.obj_loss_func(
            obj_preds_expanded,  # input: logits
            pos_mask_expanded,   # target: 0/1
            torch.ones_like(pos_mask_expanded)  # weights: all ones
        )
        # Use mean (same as original implementation)
        obj_loss = obj_loss_src.mean()
        obj_loss_weighted = obj_loss * self.obj_weight

        total_loss = reg_loss + conf_loss + obj_loss_weighted

        # Recall loss (optional, only if recall_weight > 0)
        # Use BCE loss on positive samples ONLY to encourage higher confidence predictions
        # This helps address the issue of too few detections by penalizing low confidence on GT positives
        recall_loss_weighted = 0.0
        if self.recall_weight > 0 and positives.sum() > 0:
            # Flatten obj_preds and pos_mask
            obj_preds_flat = obj_preds.view(psm.shape[0], -1)  # [B, H*W*A]
            pos_mask_flat = pos_mask.view(psm.shape[0], -1)  # [B, H*W*A]
            
            # Extract ONLY positive samples for loss calculation
            pos_indices = pos_mask_flat > 0  # [B, H*W*A]
            if pos_indices.sum() > 0:
                # Get positive samples only
                pos_obj_preds = obj_preds_flat[pos_indices]  # [N_pos]
                pos_obj_sigmoid = torch.sigmoid(pos_obj_preds)  # [N_pos]
                
                # Use simple BCE loss (not Focal Loss) for positive samples
                # Target is 1.0 for all positive samples
                recall_loss = -torch.log(pos_obj_sigmoid + 1e-6).sum()
                recall_loss_weighted = recall_loss * self.recall_weight
                total_loss = total_loss + recall_loss_weighted

        # IoU loss (optional, only if iou_weight > 0)
        iou_loss_weighted = 0.0
        if self.iou_weight > 0 and positives.sum() > 0:
            # Get anchor_box from target_dict or output_dict
            anchor_box = target_dict.get("anchor_box", None)
            if anchor_box is None:
                anchor_box = output_dict.get("anchor_box", None)
            if anchor_box is not None:
                # Use positives mask (already flattened to [B, H*W*A])
                pred_boxes = self._decode_delta_to_boxes(rm, anchor_box, positives)
                gt_boxes = self._decode_delta_to_boxes(targets, anchor_box, positives)
                
                if pred_boxes.shape[0] > 0 and gt_boxes.shape[0] > 0 and pred_boxes.shape[0] == gt_boxes.shape[0]:
                    # Calculate IoU - use mean (same as MambaFusion: sum then divide by num_pos)
                    from opencood.utils.iou3d_nms import iou3d_nms_utils
                    iou = iou3d_nms_utils.paired_boxes_iou3d_gpu(pred_boxes.float(), gt_boxes.float())
                    num_pos = pred_boxes.shape[0]
                    iou_loss = (1.0 - iou).sum() / max(num_pos, 1)
                    iou_loss_weighted = iou_loss * self.iou_weight
                    total_loss = total_loss + iou_loss_weighted

        det_loss_for_gate_debug = total_loss

        current_epoch = output_dict.get("epoch", target_dict.get("epoch", 9999))
        ############ Gate auxiliary loss ############
        if self.gate_aux_enabled and current_epoch < self.gate_warmup_epochs:
            fusion_gate_outputs = output_dict.get("fusion_gate_outputs", None)
            
            gate_aux_loss = self._compute_gate_foreground_aux_loss(
                fusion_gate_outputs=fusion_gate_outputs,
                pos_equal_one=pos_mask,
                current_epoch=current_epoch,
                gate_visibility_masks={
                    "rsu": target_dict.get("gate_visibility_extent_rsu", None),
                    "drone": target_dict.get("gate_visibility_extent_drone", None),
                }
            )
        
            if gate_aux_loss is not None:
                total_loss = total_loss + gate_aux_loss

            # ===== Gate gradient direction debug =====
            if (
                isinstance(fusion_gate_outputs, dict)
                and gate_aux_loss is not None
                and self.gate_debug_step % 50 == 0
            ):
                gate_rsu = fusion_gate_outputs.get("gate_rsu", None)
                gate_drone = fusion_gate_outputs.get("gate_drone", None)

                ref_gate = gate_rsu if isinstance(gate_rsu, torch.Tensor) else gate_drone
                if isinstance(ref_gate, torch.Tensor):
                    gate_fg_target_debug = self._build_target_from_pos_equal_one(
                        target_dict["pos_equal_one"].to(
                            device=ref_gate.device,
                            dtype=ref_gate.dtype,
                        ),
                        ref_gate.shape,
                    ).detach()

                    if isinstance(gate_rsu, torch.Tensor):
                        target_rsu = gate_fg_target_debug.to(
                            device=gate_rsu.device,
                            dtype=gate_rsu.dtype,
                        )
                        if tuple(target_rsu.shape[-2:]) != tuple(gate_rsu.shape[-2:]):
                            target_rsu = F.interpolate(
                                target_rsu,
                                size=gate_rsu.shape[-2:],
                                mode="bilinear",
                                align_corners=False,
                            )
                        visibility_rsu = self._build_visibility_mask_from_extent(
                            target_dict.get("gate_visibility_extent_rsu", None),
                            target_dict["pos_equal_one"],
                            gate_rsu.shape,
                            gate_rsu.device,
                            gate_rsu.dtype,
                        )
                        if visibility_rsu is not None:
                            target_rsu = (target_rsu * visibility_rsu).detach()
                        self._debug_gate_grad_by_region(
                            gate=gate_rsu,
                            target=target_rsu,
                            det_loss=det_loss_for_gate_debug,
                            aux_loss=gate_aux_loss,
                            name="rsu",
                        )

                    if isinstance(gate_drone, torch.Tensor):
                        target_drone = gate_fg_target_debug.to(
                            device=gate_drone.device,
                            dtype=gate_drone.dtype,
                        )
                        if tuple(target_drone.shape[-2:]) != tuple(gate_drone.shape[-2:]):
                            target_drone = F.interpolate(
                                target_drone,
                                size=gate_drone.shape[-2:],
                                mode="bilinear",
                                align_corners=False,
                            )
                        visibility_drone = self._build_visibility_mask_from_extent(
                            target_dict.get("gate_visibility_extent_drone", None),
                            target_dict["pos_equal_one"],
                            gate_drone.shape,
                            gate_drone.device,
                            gate_drone.dtype,
                        )
                        if visibility_drone is not None:
                            target_drone = (target_drone * visibility_drone).detach()
                        self._debug_gate_grad_by_region(
                            gate=gate_drone,
                            target=target_drone,
                            det_loss=det_loss_for_gate_debug,
                            aux_loss=gate_aux_loss,
                            name="drone",
                        )
            # =========================================
            self.gate_debug_step += 1
        else:
            gate_aux_loss = None

        ############ Need auxiliary loss ############
        if self.need_aux_enabled and current_epoch < self.need_aux_warmup_epochs:
            fusion_aux_outputs = output_dict.get("fusion_aux_outputs", None)
            
            need_aux_loss = self._compute_need_foreground_aux_loss(
                fusion_aux_outputs=fusion_aux_outputs,
                pos_equal_one=pos_mask,
                target_dict=target_dict,
                current_epoch=current_epoch,
            )
        
            if need_aux_loss is not None:
                total_loss = total_loss + need_aux_loss
        else:
            need_aux_loss = None

        loss_dict_update = {
                "total_loss{}".format(prefix): total_loss.item(),
                "reg_loss{}".format(prefix): reg_loss.item(),
                "conf_loss{}".format(prefix): conf_loss.item(),
            "obj_loss{}".format(prefix): obj_loss_weighted.item(),
            }
        if gate_aux_loss is not None:
            loss_dict_update[
                "gate_aux_loss{}".format(prefix)
            ] = gate_aux_loss.item()
        if need_aux_loss is not None:
            loss_dict_update[
                "fusion_need_aux_loss{}".format(prefix)
            ] = need_aux_loss.item()
        # Always record recall_loss and iou_loss if weights are set, even if 0
        if self.recall_weight > 0:
            if isinstance(recall_loss_weighted, torch.Tensor):
                loss_dict_update["recall_loss{}".format(prefix)] = recall_loss_weighted.item()
            else:
                loss_dict_update["recall_loss{}".format(prefix)] = float(recall_loss_weighted)
        if self.iou_weight > 0:
            if isinstance(iou_loss_weighted, torch.Tensor):
                loss_dict_update["iou_loss{}".format(prefix)] = iou_loss_weighted.item()
            else:
                loss_dict_update["iou_loss{}".format(prefix)] = float(iou_loss_weighted)
        self.loss_dict.update(loss_dict_update)

        return total_loss

    def cls_loss_func(
        self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor
    ):
        """
        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets
            weights: (B, #anchors) float tensor.
                Anchor-wise weights.

        Returns:
            weighted_loss: (B, #anchors, #classes) float tensor after weighting.
        """
        B, H, W, AC = input.shape
        C = self.cls_num
        A = AC // C

        input = input.view(B, H, W, A, C)        # [B, H, W, A, C]
        target = target.view(B, H, W, A, C)      # [B, H, W, A, C]
        weights = weights.view(B, H, W, A, 1)    # [B, H, W, A, 1]

        pred_sigmoid = torch.sigmoid(input)
        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        pt = target * (1.0 - pred_sigmoid) + (1.0 - target) * pred_sigmoid
        focal_weight = alpha_weight * torch.pow(pt, self.gamma)

        bce_loss = self.sigmoid_cross_entropy_with_logits(input, target)
        loss = focal_weight * bce_loss

        # Apply weights per anchor
        weighted_loss = loss * weights  # shape [B, H, W, A, C]

        return weighted_loss.sum() / B
    
    def obj_loss_func(
        self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor
    ):
        """
        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets
            weights: (B, #anchors) float tensor.
                Anchor-wise weights.

        Returns:
            weighted_loss: (B, #anchors, #classes) float tensor after weighting.
        """
        pred_sigmoid = torch.sigmoid(input)
        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        pt = target * (1.0 - pred_sigmoid) + (1.0 - target) * pred_sigmoid
        focal_weight = alpha_weight * torch.pow(pt, self.gamma)

        bce_loss = self.sigmoid_cross_entropy_with_logits(input, target)

        loss = focal_weight * bce_loss

        if weights.shape.__len__() == 2 or (
            weights.shape.__len__() == 1 and target.shape.__len__() == 2
        ):
            weights = weights.unsqueeze(-1)

        assert weights.shape.__len__() == loss.shape.__len__()

        return loss * weights

    @staticmethod
    def sigmoid_cross_entropy_with_logits(input: torch.Tensor, target: torch.Tensor):
        """PyTorch Implementation for tf.nn.sigmoid_cross_entropy_with_logits:
            max(x, 0) - x * z + log(1 + exp(-abs(x))) in
            https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets

        Returns:
            loss: (B, #anchors, #classes) float tensor.
                Sigmoid cross entropy loss without reduction
        """
        loss = (
            torch.clamp(input, min=0)
            - input * target
            + torch.log1p(torch.exp(-torch.abs(input)))
        )
        return loss

    def _decode_delta_to_boxes(self, deltas, anchor_box, pos_mask):
        """
        Decode delta (relative to anchor) to absolute boxes.
        
        Args:
            deltas: [B, H*W*A, 7] - delta values
            anchor_box: [H, W, A, 7] or [H*W*A, 7] - anchor boxes
            pos_mask: [B, H*W*A] - positive mask (bool)
            
        Returns:
            boxes: [N_pos, 7] - decoded boxes for positive samples
        """
        B, N, _ = deltas.shape
        device = deltas.device
        
        # Reshape anchor_box if needed
        if anchor_box.dim() == 4:  # [H, W, A, 7]
            anchor_box = anchor_box.view(-1, 7)  # [H*W*A, 7]
        elif anchor_box.dim() == 3:  # [H, W, A] -> should not happen
            anchor_box = anchor_box.view(-1, 7)
        
        anchor_box = anchor_box.to(device).float()
        
        # Get positive samples (flatten batch dimension)
        pos_indices_flat = pos_mask.view(-1)  # [B*H*W*A]
        if pos_indices_flat.sum() == 0:
            return torch.empty((0, 7), device=device)
        
        # Flatten deltas and get positive samples
        deltas_flat = deltas.view(-1, 7)  # [B*H*W*A, 7]
        pos_deltas = deltas_flat[pos_indices_flat]  # [N_pos, 7]
        pos_anchors = anchor_box[pos_indices_flat]  # [N_pos, 7]
        
        # Decode boxes (same logic as delta_to_boxes3d)
        boxes = torch.zeros_like(pos_deltas)
        
        # Calculate anchor diagonal for x, y normalization
        anchors_d = torch.sqrt(pos_anchors[:, 4] ** 2 + pos_anchors[:, 5] ** 2)  # [N_pos]
        
        # Decode x, y (normalized by anchor diagonal)
        boxes[:, 0] = pos_deltas[:, 0] * anchors_d + pos_anchors[:, 0]
        boxes[:, 1] = pos_deltas[:, 1] * anchors_d + pos_anchors[:, 1]
        
        # Decode z (normalized by anchor height)
        boxes[:, 2] = pos_deltas[:, 2] * pos_anchors[:, 3] + pos_anchors[:, 2]
        
        # Decode h, w, l (exp scale)
        boxes[:, 3] = torch.exp(pos_deltas[:, 3]) * pos_anchors[:, 3]  # h
        boxes[:, 4] = torch.exp(pos_deltas[:, 4]) * pos_anchors[:, 4]  # w
        boxes[:, 5] = torch.exp(pos_deltas[:, 5]) * pos_anchors[:, 5]  # l
        
        # Decode yaw (additive)
        boxes[:, 6] = pos_deltas[:, 6] + pos_anchors[:, 6]
        
        return boxes

    @staticmethod
    def add_sin_difference(boxes1, boxes2, dim=6):
        assert dim != -1
        rad_pred_encoding = torch.sin(boxes1[..., dim : dim + 1]) * torch.cos(
            boxes2[..., dim : dim + 1]
        )
        rad_tg_encoding = torch.cos(boxes1[..., dim : dim + 1]) * torch.sin(
            boxes2[..., dim : dim + 1]
        )

        boxes1 = torch.cat(
            [boxes1[..., :dim], rad_pred_encoding, boxes1[..., dim + 1 :]], dim=-1
        )
        boxes2 = torch.cat(
            [boxes2[..., :dim], rad_tg_encoding, boxes2[..., dim + 1 :]], dim=-1
        )
        return boxes1, boxes2

    @staticmethod
    def smooth_l1_loss(diff, beta):
        if beta < 1e-5:
            loss = torch.abs(diff)
        else:
            n = torch.abs(diff)
            loss = torch.where(n < beta, 0.5 * n**2 / beta, n - 0.5 * beta)
        return loss

    def logging(self, epoch, batch_id, batch_len, writer=None):
        """
        Print out  the loss function for current iteration.

        Parameters
        ----------
        epoch : int
            Current epoch for training.
        batch_id : int
            The current batch.
        batch_len : int
            Total batch length in one iteration of training,
        writer : SummaryWriter
            Used to visualize on tensorboard
        """
        total_loss = [v for k, v in self.loss_dict.items() if "total_loss" in k]
        if len(total_loss) > 1:
            total_loss = sum(total_loss)
        else:
            total_loss = total_loss[0]

        print_msg = "[epoch {}][{}/{}], || Loss: {:.2f} ||".format(
            epoch, batch_id + 1, batch_len, total_loss
        )
        for k, v in self.loss_dict.items():
            print_msg += "{}: {:.2f} | ".format(
                k.replace("_loss", "").replace("_single", ""), v
            )

        if not writer is None:
            for k, v in self.loss_dict.items():
                writer.add_scalar(k, v, epoch * batch_len + batch_id)
                
        return print_msg

    def _forward(self, output_dict, target_dict, prefix=""):
        """
        Parameters
        ----------
        output_dict : dict
        target_dict : dict

        cls_label -> one_hot label
        """
        rm = output_dict["rm{}".format(prefix)]  # [B, 14, 50, 176]
        psm = output_dict["psm{}".format(prefix)]  # [B, 2, 50, 176]
        targets = target_dict["targets"]

        cls_preds = psm.permute(0, 2, 3, 1).contiguous()  # N, C, H, W -> N, H, W, C

        box_cls_labels = target_dict["pos_equal_one"]  # [B, 50, 176, 2]
        box_cls_labels = box_cls_labels.view(psm.shape[0], -1).contiguous()

        positives = box_cls_labels > 0
        negatives = box_cls_labels == 0
        negative_cls_weights = negatives * 1.0
        cls_weights = (negative_cls_weights + 1.0 * positives).float()
        reg_weights = positives.float()

        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_targets = box_cls_labels
        one_hot_targets = torch.zeros(
            *list(cls_targets.shape),
            2,
            dtype=cls_preds.dtype,
            device=cls_targets.device,
        )
        one_hot_targets.scatter_(-1, cls_targets.unsqueeze(dim=-1).long(), 1.0)
        cls_preds = cls_preds.view(psm.shape[0], -1, 1)
        one_hot_targets = one_hot_targets[..., 1:] # here remove the BG class

        cls_loss_src = self.cls_loss_func(
            cls_preds, one_hot_targets, weights=cls_weights
        )  # [N, M]
        cls_loss = cls_loss_src.sum() / psm.shape[0]
        conf_loss = cls_loss * self.cls_weight

        # regression
        rm = rm.permute(0, 2, 3, 1).contiguous()
        rm = rm.view(rm.size(0), -1, 7)
        targets = targets.view(targets.size(0), -1, 7)
        box_preds_sin, reg_targets_sin = self.add_sin_difference(rm, targets)
        loc_loss_src = self.reg_loss_func(
            box_preds_sin, reg_targets_sin, weights=reg_weights
        )

        reg_loss = loc_loss_src.sum() / rm.shape[0]
        reg_loss *= self.reg_coe

        total_loss = conf_loss
        # total_loss = reg_loss + conf_loss

        # print('psm: ', psm.shape, cls_preds.shape)
        # print('rm: ', rm.shape, box_preds_sin.shape)

        self.loss_dict.update(
            {
                "total_loss{}".format(prefix): total_loss,
                #'reg_loss{}'.format(prefix): reg_loss,
                "conf_loss{}".format(prefix): conf_loss,
            }
        )

        return total_loss

    def _logging(self, epoch, batch_id, batch_len, writer=None):
        """
        Print out  the loss function for current iteration.

        Parameters
        ----------
        epoch : int
            Current epoch for training.
        batch_id : int
            The current batch.
        batch_len : int
            Total batch length in one iteration of training,
        writer : SummaryWriter
            Used to visualize on tensorboard
        """
        total_loss = [v.item() for k, v in self.loss_dict.items() if "total_loss" in k]
        if len(total_loss) > 1:
            total_loss = sum(total_loss)
        else:
            total_loss = total_loss[0]
        # reg_loss = self.loss_dict['reg_loss']
        conf_loss = self.loss_dict["conf_loss"]

        print_msg = "[epoch {}][{}/{}], || Loss: {:.2f} ||".format(
            epoch, batch_id + 1, batch_len, total_loss
        )
        for k, v in self.loss_dict.items():
            print_msg += "{}: {:.2f} | ".format(
                k.replace("_loss", "").replace("_single", ""), v.item()
            )

        # print_msg = ("[epoch %d][%d/%d], || Loss: %.4f || Conf Loss: %.4f"
        #             " || Loc Loss: %.4f" % (
        #                 epoch, batch_id + 1, batch_len,
        #                 total_loss.item(), conf_loss.item(), reg_loss.item()))

        if self.use_dir:
            dir_loss = self.loss_dict["dir_loss"]
            print_msg += " || Dir Loss: %.4f" % dir_loss.item()

        # print(print_msg)

        if not writer is None:
            for k, v in self.loss_dict.items():
                writer.add_scalar(k, v.item(), epoch * batch_len + batch_id)
            # writer.add_scalar('Regression_loss', reg_loss.item(),
            #                 epoch*batch_len + batch_id)
            # writer.add_scalar('Confidence_loss', conf_loss.item(),
            #                 epoch*batch_len + batch_id)

            if self.use_dir:
                writer.add_scalar(
                    "dir_loss", dir_loss.item(), epoch * batch_len + batch_id
                )
        
        return print_msg