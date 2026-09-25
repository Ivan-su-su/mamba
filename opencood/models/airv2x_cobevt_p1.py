from opencood.models.airv2x_cobevt import Airv2xCoBEVT
from opencood.models.lss_pretrain_modules.p1_mixin import P1CamMixin


class Airv2xCoBEVTP1(P1CamMixin, Airv2xCoBEVT):
    """CoBEVT with frozen P1 image/depth. Fusion code unchanged."""
