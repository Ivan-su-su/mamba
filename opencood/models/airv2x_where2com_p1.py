from opencood.models.airv2x_where2com import Airv2xWhere2com
from opencood.models.lss_pretrain_modules.p1_mixin import P1CamMixin


class Airv2xWhere2comP1(P1CamMixin, Airv2xWhere2com):
    """Where2comm with frozen P1 image/depth. Fusion code unchanged."""
