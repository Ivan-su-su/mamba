"""Smoke test: 多模态 HEAL/STAMP 模型构建与通道维度验证.

仅验证：
1. YAML 能被 yaml_utils 正确加载
2. 模型能成功构建
3. _fused_encoder_channels 返回 128 (cam+lidar concat)
4. backbone 输入通道与 encoder 输出一致
5. 单模态 yaml 仍返回 64 (回归测试)

不跑真实数据前向，避免依赖完整数据集。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# 把 opencood 加入 sys.path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils


class _DummyOpt:
    """load_airv2x_params 需要的 opt 占位."""

    def __init__(self) -> None:
        self.model_dir = ""
        self.vehicle_dir = None
        self.rsu_dir = None
        self.drone_dir = None
        self.vehicle_epoch = 20
        self.rsu_epoch = 20
        self.drone_epoch = 20


def _check_yaml(name: str, yaml_path: str, expected_channels: int) -> None:
    print(f"\n=== {name}: {yaml_path} ===")
    assert os.path.exists(yaml_path), f"yaml 不存在: {yaml_path}"

    opt = _DummyOpt()
    hypes = yaml_utils.load_yaml(yaml_path, opt)
    print(f"  active_sensors: {hypes['active_sensors']}")
    print(f"  model.core_method: {hypes['model']['core_method']}")

    model = train_utils.create_model(hypes)
    print(f"  model class: {type(model).__name__}")

    # 验证 _fused_encoder_channels
    assert hasattr(model, "_fused_encoder_channels"), \
        f"{name} 未实现 _fused_encoder_channels"
    assert hasattr(model, "fuse_bev"), f"{name} 未覆盖 fuse_bev"
    assert hasattr(model, "encoder_out_channels"), \
        f"{name} 未设置 encoder_out_channels"

    got = model.encoder_out_channels
    print(f"  encoder_out_channels: got={got}, expected={expected_channels}")
    assert got == expected_channels, \
        f"{name} 通道数不符: got={got}, expected={expected_channels}"

    # 验证 backbone 第一层输入通道
    # ResNetBEVBackbone 用 ResNetModified; 注意 ResNetModified.inplanes 在
    # 构造末尾被覆写为最后一层输出通道，真正的输入通道在 layer0[0].conv1.
    layer0 = model.backbone.resnet.layer0[0]
    in_ch = layer0.conv1.in_channels
    print(f"  backbone.resnet.layer0[0].conv1.in_channels: {in_ch}")
    assert in_ch == expected_channels, \
        f"{name} backbone input channels={in_ch} != expected={expected_channels}"

    # 参数量
    total = sum(p.nelement() for p in model.parameters())
    print(f"  #params: {total/1e6:.3f}M")
    print(f"  [OK] {name}")


def main() -> None:
    base = ROOT / "opencood" / "hypes_yaml" / "airv2x"

    # 多模态 HEAL single (vehicle): cam+lidar -> 128
    _check_yaml(
        "HEAL vehicle cam_lidar (multi-modal)",
        str(base / "camera_lidar" / "det" / "airv2x_heal" / "single"
            / "airv2x_HEAL_vehicle_cam_lidar.yaml"),
        expected_channels=128,
    )

    # 多模态 HEAL collab: cam+lidar -> 128
    _check_yaml(
        "HEAL collab cam_lidar (multi-modal)",
        str(base / "camera_lidar" / "det" / "airv2x_heal"
            / "airv2x_HEAL_collab_cam_lidar.yaml"),
        expected_channels=128,
    )

    # 多模态 STAMP collab: cam+lidar -> 128
    _check_yaml(
        "STAMP collab cam_lidar (multi-modal)",
        str(base / "camera_lidar" / "det" / "airv2x_stamp"
            / "airv2x_stamp_collab_cam_lidar.yaml"),
        expected_channels=128,
    )

    # 回归测试: 单模态 HEAL collab lidar -> 64
    _check_yaml(
        "HEAL collab lidar (single-modal, regression)",
        str(base / "lidar" / "det" / "airv2x_heal"
            / "airv2x_HEAL_collab_lidar.yaml"),
        expected_channels=64,
    )

    # 回归测试: 单模态 STAMP collab lidar -> 64
    _check_yaml(
        "STAMP collab lidar (single-modal, regression)",
        str(base / "lidar" / "det" / "airv2x_stamp"
            / "airv2x_stamp_collab_lidar.yaml"),
        expected_channels=64,
    )

    print("\n所有 smoke 测试通过 ✅")


if __name__ == "__main__":
    main()
