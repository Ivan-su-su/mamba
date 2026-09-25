from opencood.models.airv2x_v2xvit import Airv2xV2XVit
from opencood.models.lss_pretrain_modules.p1_mixin import P1CamMixin


class Airv2xV2XVitP1(P1CamMixin, Airv2xV2XVit):
    """V2X-ViT with frozen P1 image/depth. Fusion code unchanged."""
