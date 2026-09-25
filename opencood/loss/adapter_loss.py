# -*- coding: utf-8 -*-
# Author: Xiangbo Gao <xiangbogaobarry@gmail.com>
# License: MIT License

from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class AdapterLoss(nn.Module):
    """STAMP feature-alignment loss (P2M / M2P2M / M2P).

    Supports:
    1. Classic 5-tensor call: ``(FM, FP2M, FM2P2M, FP, FM2P)``
    2. Dict call from ``airv2x_stamp``: ``adapter_align`` with per-agent
       ``FM / FM2P / FM2P2M``. Vehicle local features act as protocol when
       present; otherwise only the cycle term ``M2P2M`` is used.
    """

    def __init__(self, args: Dict[str, Any]) -> None:
        """Initialize weighted MSE alignment loss.

        Args:
            args: Config with ``alpha_P2M``, ``alpha_M2P2M``, ``alpha_M2P``.
        """
        super(AdapterLoss, self).__init__()
        self.alpha_P2M = args.get("alpha_P2M", 1.0)
        self.alpha_M2P2M = args.get("alpha_M2P2M", 1.0)
        self.alpha_M2P = args.get("alpha_M2P", 1.0)
        self.l2loss = nn.MSELoss()
        self.loss_dict: Dict[str, torch.Tensor] = {}

    def forward(
        self,
        FM: Any,
        FP2M: Optional[torch.Tensor] = None,
        FM2P2M: Optional[torch.Tensor] = None,
        FP: Optional[torch.Tensor] = None,
        FM2P: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute adapter alignment loss.

        Args:
            FM: Local feature tensor, or ``adapter_align`` dict from the model.
            FP2M: Protocol→local features (classic API).
            FM2P2M: Cycle-reconstructed local features (classic API).
            FP: Protocol features (classic API).
            FM2P: Local→protocol features (classic API).

        Returns:
            Weighted sum of available alignment terms.
        """
        if isinstance(FM, dict):
            return self._forward_from_align_dict(FM)

        assert FP2M is not None and FM2P2M is not None
        assert FP is not None and FM2P is not None
        p2m = self.l2loss(FM, FP2M)
        m2p2m = self.l2loss(FM, FM2P2M)
        m2p = self.l2loss(FP, FM2P)
        total_loss = (
            self.alpha_P2M * p2m + self.alpha_M2P2M * m2p2m + self.alpha_M2P * m2p
        )
        self.loss_dict.update(
            {"total_loss": total_loss, "P2M": p2m, "M2P2M": m2p2m, "M2P": m2p}
        )
        return total_loss

    def _forward_from_align_dict(
        self, align_feats: Dict[str, Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """Aggregate per-agent cycle loss; use vehicle as protocol when present."""
        device = next(iter(next(iter(align_feats.values())).values())).device
        zero = torch.tensor(0.0, device=device)
        p2m_sum = zero
        m2p_sum = zero
        m2p2m_sum = zero
        n_cycle = 0
        n_proto = 0

        protocol_fp: Optional[torch.Tensor] = None
        if "vehicle" in align_feats:
            # Vehicle domain is the protocol domain for this minimal patch.
            protocol_fp = align_feats["vehicle"]["FM"].detach()

        for agent, feats in align_feats.items():
            fm = feats["FM"]
            fm2p = feats["FM2P"]
            fm2p2m = feats["FM2P2M"]

            m2p2m_sum = m2p2m_sum + self.l2loss(fm, fm2p2m)
            n_cycle += 1

            if protocol_fp is not None and agent == "vehicle":
                # Vehicle domain == protocol: push adapter/reverter toward identity.
                m2p_sum = m2p_sum + self.l2loss(protocol_fp, fm2p)
                p2m_sum = p2m_sum + self.l2loss(fm, feats["FP2M"])
                n_proto += 1

        m2p2m = m2p2m_sum / max(n_cycle, 1)
        m2p = m2p_sum / max(n_proto, 1) if n_proto > 0 else zero
        p2m = p2m_sum / max(n_proto, 1) if n_proto > 0 else zero

        total_loss = (
            self.alpha_P2M * p2m + self.alpha_M2P2M * m2p2m + self.alpha_M2P * m2p
        )
        self.loss_dict.update(
            {"total_loss": total_loss, "P2M": p2m, "M2P2M": m2p2m, "M2P": m2p}
        )
        return total_loss

    def logging(
        self,
        epoch: int,
        batch_id: int,
        batch_len: int,
        writer: Any = None,
    ) -> str:
        """Log adapter losses to stdout / tensorboard."""
        total_loss = self.loss_dict["total_loss"]
        p2m_loss = self.loss_dict["P2M"]
        m2p2m_loss = self.loss_dict["M2P2M"]
        m2p_loss = self.loss_dict["M2P"]

        msg = (
            "[epoch %d][%d/%d], || Adapter Loss: %.6f || P2M: %.6f"
            " || M2P2M: %.6f || M2P: %.6f"
            % (
                epoch,
                batch_id + 1,
                batch_len,
                total_loss.item(),
                p2m_loss.item(),
                m2p2m_loss.item(),
                m2p_loss.item(),
            )
        )
        print(msg)

        if writer is not None:
            step = epoch * batch_len + batch_id
            writer.add_scalar("P2M_loss", p2m_loss.item(), step)
            writer.add_scalar("M2P2M_loss", m2p2m_loss.item(), step)
            writer.add_scalar("M2P_loss", m2p_loss.item(), step)
        return msg
