from .mambafusion import MambaFusion
from .mambafusion_all import MambaFusion_all
from .mambafusion_new import MambaFusion_new
__all__ = {
    'MambaFusion_all': MambaFusion_all,   #将所有voxel投影到所有图像的错误版本
    'MambaFusion': MambaFusion,   #原始版本
    'MambaFusion_new': MambaFusion_new   #将某agent的voxel只投影到该agent的图像，且加入voxel embedding
}

