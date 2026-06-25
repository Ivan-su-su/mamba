import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def make_cuda_ext(name, module, sources):
    return CUDAExtension(
        name=f"{module}.{name}",
        sources=[os.path.join(*module.split("."), src) for src in sources],
    )


setup(
    name="airv2x_bev_pool_ops",
    cmdclass={"build_ext": BuildExtension},
    ext_modules=[
        make_cuda_ext(
            name="bev_pool_ext",
            module="opencood.models.mambafusion_modules.ops.bev_pool",
            sources=[
                "src/bev_pool.cpp",
                "src/bev_pool_cuda.cu",
            ],
        ),
        make_cuda_ext(
            name="bev_pool_v2_ext",
            module="opencood.models.mambafusion_modules.ops.bev_pool_v2",
            sources=[
                "src/bev_pool.cpp",
                "src/bev_pool_cuda.cu",
            ],
        ),
    ],
)
